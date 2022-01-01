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

RAFT_Torch::RAFT_Torch(){
    raft =std::make_unique<torch::jit::Module>(torch::jit::load("/home/chen/PycharmProjects/RAFT/kitti.pt"));
}


vector<Tensor> RAFT_Torch::forward(Tensor &tensor0, Tensor &tensor1) {
    auto result = raft->forward({tensor0,tensor1}).toTensorVector();
    return result;
}

