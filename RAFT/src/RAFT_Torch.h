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


class RAFT_Torch {
public:
    using Ptr = std::unique_ptr<RAFT_Torch>;

    RAFT_Torch();
    vector<Tensor> forward(Tensor& tensor0, Tensor& tensor1);


private:
    std::shared_ptr<torch::jit::Module> raft;

};


#endif //RAFT_CPP_RAFT_TORCH_H
