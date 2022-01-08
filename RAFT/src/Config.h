/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of RAFT_Libtorch.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef RAFT_CPP_CONFIG_H
#define RAFT_CPP_CONFIG_H



#include <vector>
#include <fstream>
#include <map>
#include <iostream>
#include <exception>
#include <tuple>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::pair;
using std::vector;
using std::tuple;

using namespace std::chrono_literals;
namespace fs=std::filesystem;



inline std::shared_ptr<spdlog::logger> sgLogger;

template <typename Arg1, typename... Args>
inline void Debugs(const char* fmt, const Arg1 &arg1, const Args&... args){ sgLogger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void Debugs(const T& msg){sgLogger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void Infos(const char* fmt, const Arg1 &arg1, const Args&... args){sgLogger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void Infos(const T& msg){sgLogger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void Warns(const char* fmt, const Arg1 &arg1, const Args&... args){sgLogger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void Warns(const T& msg){sgLogger->log(spdlog::level::warn, msg);}


class Config {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr=std::shared_ptr<Config>;

    explicit Config(const std::string &file_name);

    inline static std::string model_path;
    inline static int imageH,imageW;

    inline static std::string SEGMENTOR_LOG_PATH;
    inline static std::string SEGMENTOR_LOG_LEVEL;
    inline static std::string SEGMENTOR_LOG_FLUSH;

    inline static std::atomic_bool ok{true};

    inline static string DATASET_DIR;
    inline static string WARN_UP_IMAGE_PATH;
};


#endif //RAFT_CPP_CONFIG_H
