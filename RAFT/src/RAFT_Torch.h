/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of RAFT_Libtorch.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef RAFT_CPP_RAFT_TORCH_H
#define RAFT_CPP_RAFT_TORCH_H

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "Config.h"

using Tensor = torch::Tensor;


class RAFT_TorchScript {
public:
    using Ptr = std::unique_ptr<RAFT_TorchScript>;
    RAFT_TorchScript();
    vector<Tensor> Forward(Tensor& tensor0, Tensor& tensor1);
private:
    std::shared_ptr<torch::jit::Module> raft;
};



class RAFT_Torch{
public:
    using Ptr = std::unique_ptr<RAFT_Torch>;

    RAFT_Torch();

    vector<Tensor> Forward(Tensor& tensor0, Tensor& tensor1);

private:
    tuple<Tensor,Tensor> ForwardFnet(Tensor &tensor0, Tensor &tensor1);
    tuple<Tensor,Tensor> ForwardCnet(Tensor &tensor);
    tuple<Tensor,Tensor,Tensor> ForwardUpdate(Tensor &net, Tensor &inp, Tensor &corr, Tensor &flow);
    static tuple<Tensor,Tensor> InitializeFlow(Tensor &tensor);
    void ComputeCorrPyramid(Tensor &tensor0, Tensor &tensor1);
    Tensor IndexCorrVolume(Tensor &tensor);

    Tensor last_flow;
    vector<Tensor> corr_pyramid; //相关性金字塔

    std::shared_ptr<torch::jit::Module> fnet_;
    std::shared_ptr<torch::jit::Module> cnet_;
    std::shared_ptr<torch::jit::Module> update_;
};




#endif //RAFT_CPP_RAFT_TORCH_H
