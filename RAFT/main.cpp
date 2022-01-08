/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of RAFT_Libtorch.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <iostream>
#include <memory>
#include <filesystem>
#include "src/Config.h"
#include "src/utils.h"
#include "src/Pipeline.h"
#include "src/Visualization.h"
#include "src/RAFT_Torch.h"

namespace fs = std::filesystem;

cv::Mat ReadOneKitti(int index){
    char name[64];
    sprintf(name,"%06d.png",index);
    std::string img0_path=Config::DATASET_DIR+name;
    fmt::print("Read Image:{}\n",img0_path);
    return cv::imread(img0_path);
}


vector<fs::path> ReadImagesNames(const string &path){
    fs::path dir_path(path);
    vector<fs::path> names;
    if(!fs::exists(dir_path))
        return names;
    fs::directory_iterator dir_iter(dir_path);
    for(auto &it : dir_iter){
        names.push_back(it.path());
    }
    return names;
}


int main(int argc, char **argv) {
    if(argc != 2){
        cerr<<"please input: [config file]"<<endl;
        return 1;
    }
    string config_file = argv[1];
    fmt::print("config_file:{}\n",argv[1]);
    //RAFT_TorchScript::Ptr raft_torchscript;
    RAFT_Torch::Ptr raft_torch;
    try{
        Config cfg(config_file);
        raft_torch = std::make_unique<RAFT_Torch>();
    }
    catch(std::runtime_error &e){
        sgLogger->critical(e.what());
        cerr<<e.what()<<endl;
        return -1;
    }

    TicToc tt;

    auto names = ReadImagesNames(Config::DATASET_DIR);
    std::sort(names.begin(),names.end());

    cv::Mat img0,img1;
    img0 = cv::imread(names[0].string());


    Tensor flow;
    Tensor tensor0 = Pipeline::process(img0);//(1,3,376, 1232),值大小从-1到1


    for(int index=1; index <1000;++index)
    {
        img1 = cv::imread(names[index].string());
        fmt::print(names[index].string()+"\n");
        if(img1.empty()){
            cerr<<"Read image:"<<index<<" failure"<<endl;
            break;
        }
        tt.Tic();
        Tensor tensor1 = Pipeline::process(img1);//(1,3,376, 1232)

        Debugs("process:{} ms", tt.TocThenTic());

        vector<Tensor> prediction = raft_torch->Forward(tensor0, tensor1);

        Debugs("prediction:{} ms", tt.TocThenTic());
/*
        torch::Tensor tensor1_raw = (tensor1.squeeze()+1.)/2.;
        flow = prediction.back();//[1,2,h,w]
        flow = flow.squeeze();

        cv::Mat flow_show = visual_flow_image(tensor1_raw,flow);
        //cv::Mat flow_show = visual_flow_image(flow);

        cv::imshow("flow",flow_show);
        if(auto order=(cv::waitKey(100) & 0xFF); order == 'q')
            break;
        else if(order==' ')
            cv::waitKey(0);*/

        tensor0 = tensor1;
    }


    return 0;
}
