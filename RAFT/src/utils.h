/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of RAFT_Libtorch.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef RAFT_CPP_UTILS_H
#define RAFT_CPP_UTILS_H

#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <random>

#include <torch/torch.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <spdlog/logger.h>

#include <NvInfer.h>



class TicToc{
public:
    TicToc(){
        Tic();
    }

    void Tic(){
        start_ = std::chrono::system_clock::now();
    }

    double Toc(){
        end_ = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_ - start_;
        return elapsed_seconds.count() * 1000;
    }

    double TocThenTic(){
        auto t= Toc();
        Tic();
        return t;
    }

    void TocPrintTic(const char* str){
        std::cout << str << ":" << Toc() << " ms" << std::endl;
        Tic();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start_, end_;
};


class utils {

};



template <typename T>
static std::string DimsToStr(torch::ArrayRef<T> list){
    int i = 0;
    std::string text= "[";
    for(auto e : list) {
        if (i++ > 0) text+= ", ";
        text += std::to_string(e);
    }
    text += "]";
    return text;
}


static std::string DimsToStr(nvinfer1::Dims list){
    std::string text= "[";
    for(int i=0;i<list.nbDims;++i){
        if (i > 0) text+= ", ";
        text += std::to_string(list.d[i]);
    }
    text += "]";
    return text;
}



inline cv::Point2f operator*(const cv::Point2f &lp,const cv::Point2f &rp)
{
    return {lp.x * rp.x,lp.y * rp.y};
}

template<typename MatrixType>
inline std::string EigenToStr(const MatrixType &m){
    std::string text;
    for(int i=0;i<m.rows();++i){
        for(int j=0;j<m.cols();++j){
            text+=fmt::format("{:.2f} ",m(i,j));
        }
        if(m.rows()>1)
            text+="\n";
    }
    return text;
}


template<typename T>
inline std::string VecToStr(const Eigen::Matrix<T,3,1> &vec){
    return EigenToStr(vec.transpose());
}


inline cv::Scalar_<unsigned int> getRandomColor(){
    static std::default_random_engine rde;
    static std::uniform_int_distribution<unsigned int> color_rd(0,255);
    return {color_rd(rde),color_rd(rde),color_rd(rde)};
}



#endif //RAFT_CPP_UTILS_H
