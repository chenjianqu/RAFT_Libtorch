/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of RAFT_Libtorch.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef RAFT_CPP_PIPELINE_H
#define RAFT_CPP_PIPELINE_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

class Pipeline {
public:


    static torch::Tensor process(cv::Mat &img);


private:

};


#endif //RAFT_CPP_PIPELINE_H
