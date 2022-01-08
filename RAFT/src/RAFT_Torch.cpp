/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of RAFT_Libtorch.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "RAFT_Torch.h"
#include <torch/script.h>
#include "utils.h"

namespace F=torch::nn::functional;

RAFT_TorchScript::RAFT_TorchScript(){
    raft =std::make_unique<torch::jit::Module>(torch::jit::load("/home/chen/PycharmProjects/RAFT/kitti.pt"));
}


vector<Tensor> RAFT_TorchScript::Forward(Tensor &tensor0, Tensor &tensor1) {
    auto result = raft->forward({tensor0,tensor1}).toTensorVector();
    return result;
}




RAFT_Torch::RAFT_Torch(){
    fnet_ =std::make_unique<torch::jit::Module>(torch::jit::load(
            "/home/chen/CLionProjects/RAFT_Libtorch/weights/viode_fnet.pth"));
    cnet_ =std::make_unique<torch::jit::Module>(torch::jit::load(
            "/home/chen/CLionProjects/RAFT_Libtorch/weights/viode_cnet.pth"));
    update_ =std::make_unique<torch::jit::Module>(torch::jit::load(
            "/home/chen/CLionProjects/RAFT_Libtorch/weights/viode_update.pth"));

}

tuple<Tensor, Tensor> RAFT_Torch::ForwardFnet(Tensor &tensor0, Tensor &tensor1) {
    auto result = fnet_->forward({tensor0,tensor1}).toTuple();
    return {result->elements()[0].toTensor(),result->elements()[1].toTensor()};
}


tuple<Tensor, Tensor> RAFT_Torch::ForwardCnet(Tensor &tensor) {
    auto result = cnet_->forward({tensor}).toTensor();
    auto t_vector = torch::split_with_sizes(result,{128,128},1);
    auto net = torch::tanh(t_vector[0]);//[1,128,47,154]
    auto inp = torch::relu(t_vector[1]);
    return {net,inp};
}


tuple<Tensor, Tensor, Tensor> RAFT_Torch::ForwardUpdate(Tensor &net, Tensor &inp, Tensor &corr, Tensor &flow) {
    auto result = update_->forward({net,inp,corr,flow}).toTuple();
    return {result->elements()[0].toTensor(),result->elements()[1].toTensor(),result->elements()[2].toTensor()};
}

/**
 * 计算相关金字塔
 * @param tensor0
 * @param tensor1
 * @return
 */
void RAFT_Torch::ComputeCorrPyramid(Tensor &tensor0, Tensor &tensor1) {
    auto size = tensor0.sizes();
    const int batch = size[0];
    const int dim = size[1];
    const int h = size[2];
    const int w = size[3];
    const int num_level = 4;
    corr_pyramid.clear();

    ///计算corr张量
    Tensor t0_view = tensor0.view({batch,dim,h*w});//[1, 256, 47*154]
    Tensor t1_view = tensor1.view({batch,dim,h*w});//[1, 256, 47*154]
    Tensor corr = torch::matmul(t0_view.transpose(1,2),t1_view);//[1, 7238,7238]
    corr = corr.view({batch,h,w,1,h,w});//[1,47,154,1,47,154]
    corr = corr / std::sqrt(dim);

    corr = corr.reshape({batch*h*w,1,h,w}).to(torch::kFloat);

    ///构造corr volume金字塔
    corr_pyramid.push_back(corr);//[batch*h*w,1,h,w]
    static auto opt =F::AvgPool2dFuncOptions(2).stride(2);//kernel size =2 ,stride =2;

    for(int i=1;i<num_level;++i){
        corr = F::avg_pool2d(corr,opt);
        corr_pyramid.push_back(corr);
    }
}

/**
 * 初始化光流
 * @param tensor [1,3,376, 1232]
 * @return {coords0,coords1},shape:[1,2,47,154]
 */
tuple<Tensor, Tensor> RAFT_Torch::InitializeFlow(Tensor &tensor) {
    auto size = tensor.sizes();
    static auto opt = torch::TensorOptions(torch::kCUDA);
    auto coords_grid = [](int batch,int h,int w){
        auto coords_vector = torch::meshgrid({torch::arange(h,opt),torch::arange(w,opt)});//[h,w]
        auto coords = torch::stack({coords_vector[1],coords_vector[0]},0).to(torch::kFloat);//[2,h,w]
        return coords.unsqueeze(0).expand({batch,2,h,w});//(1,2,h,w)
    };

    auto coords0 = coords_grid(size[0],size[2]/8,size[3]/8);
    auto coords1 = coords0.clone();
    return {coords0,coords1};
}

/**
 * 索引相关性张量
 * @param tensor [1,2,47,154]
 * @param pyramid
 * @return [batch*h*w,1,9,9]
 */
Tensor RAFT_Torch::IndexCorrVolume(Tensor &tensor){
    auto bilinearSampler = [](Tensor &img,Tensor &coords){ //img:[7238,1,h,w], coords:[7238, 9, 9, 2]
        int H = img.sizes()[2];
        int W = img.sizes()[3];
        auto grids = coords.split_with_sizes({1,1},-1);//划分为两个[7238,9,9,1]的张量
        Tensor xgrid = 2*grids[0]/(W-1) -1;//归一化到[-1,1]
        Tensor ygrid = 2*grids[1]/(H-1) -1;
        Tensor grid = torch::cat({xgrid,ygrid},-1);//[7238, 9, 9, 2]
        static auto opt = F::GridSampleFuncOptions().align_corners(true);
        return grid_sample(img,grid,opt);
    };

    static auto gpu = torch::TensorOptions(torch::kCUDA);

    const int r = 4;
    const int rr = 2*r+1;
    const int num_level = 4;
    auto coords = tensor.permute({0,2,3,1});//[batch,h,w,2]
    auto size = coords.sizes();
    const int batch = size[0], h = size[1],w = size[2];

    vector<Tensor> out_pyramid;
    for(int i=0;i<num_level;++i){
        auto corr = corr_pyramid[i];//层i的相关性张量，[7238,1,h,w]
        //每个像素在该金字塔层搜索的范围
        auto delta = torch::stack(torch::meshgrid( //[9,9,2],  [2*r+1, 2*r+1,2]
                {torch::linspace(-r,r,rr,gpu),torch::linspace(-r,r,rr,gpu)}),-1);
        auto delta_lvl = delta.view({1,rr,rr,2});//[1,9,9,2]
        //将坐标值缩放到层i的值
        auto centroid_lvl = coords.reshape({batch*h*w,1,1,2}) / std::pow(2,i);//[batch*h*w,1,1,2]
        auto coords_lvl = centroid_lvl + delta_lvl;//[7238, 9, 9, 2]，7238表示每个像素，9 9表示每个像素检索的范围，2表示xy值
        corr =bilinearSampler(corr,coords_lvl);//[batch*h*w,1,9,9]
        corr = corr.view({batch,h,w,-1});//[batch,h,w,81]
        out_pyramid.push_back(corr);
    }
    Tensor out = torch::cat(out_pyramid,-1);
    return out.permute({0,3,1,2}).contiguous().to(torch::kFloat);
}





/**
 * 光流估计
 * @param tensor0 图像1 [1,3,376, 1232]
 * @param tensor1 图像2 [1,3,376, 1232]
 * @return
 */
vector<Tensor> RAFT_Torch::Forward(Tensor& tensor0, Tensor& tensor1) {
    TicToc tt;
    const int num_iter = 20;

    auto [fmat0,fmat1] = ForwardFnet(tensor0, tensor1);//fmat0和fmat1:[1, 256, 47, 154]
    Debugs("ForwardFnet:{} ms", tt.TocThenTic());

    /**
     * [7238, 1, 47, 154]
     * [7238, 1, 23, 77]
     * [7238, 1, 11, 38]
     * [7238, 1, 5, 19]
     */
    ComputeCorrPyramid(fmat0, fmat1);
    Debugs("corr_pyramid:{} ms", tt.TocThenTic());
    //for(auto &p : corr_pyramid) debug_s("corr_pyramid.shape:{}", dims2str(p.sizes()));
    auto [net,inp] = ForwardCnet(tensor1);//net和inp:[1,128,47,154]
    Debugs("ForwardCnet:{} ms", tt.TocThenTic());

    auto [coords0,coords1] = InitializeFlow(tensor1);//coords0和coords1：[1,2,47,154]
    Debugs("InitializeFlow:{} ms", tt.TocThenTic());

    if(last_flow.defined())coords1 = coords1 + last_flow;

    vector<Tensor> flow_prediction;
    for(int i=0;i<num_iter;++i){
        Debugs("{}", i);
        auto corr = IndexCorrVolume(coords1);//[batch*h*w, 1, 9, 9]
        auto flow = coords1 - coords0;//[1,2,47,154]
        auto [net1,up_mask,delta_flow] = ForwardUpdate(net, inp, corr, flow);
        net = net1;
        coords1 = coords1 + delta_flow;

        ///terminate called after throwing an instance of 'c10::CUDAOutOfMemoryError'
        if(i==num_iter-1){///上采样
            flow = coords1 - coords0;
            static auto opt = F::InterpolateFuncOptions().size(
                    vector<int64_t>({8*coords1.sizes()[2],8*coords1.sizes()[3]})).
                            mode(torch::kBilinear).align_corners(true);
            auto flow_up = 8 * interpolate(flow,opt) ;//[1, 2, 376, 1232]
            flow_prediction.push_back(flow_up);
            //last_flow =  coords1 - coords0;
            //Debugs("last_flow:{}", DimsToStr(last_flow.sizes()));
        }
    }
    Debugs("iter all:{} ms", tt.TocThenTic());

    corr_pyramid.clear();


    return flow_prediction;
}


