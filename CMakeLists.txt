cmake_minimum_required(VERSION 3.20)
project(RAFT_Libtorch)

set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


#Libtorch
set(CMAKE_PREFIX_PATH "/usr/local/lib;/home/chen/app/libtorch")

find_package(TorchVision REQUIRED)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

aux_source_directory(RAFT/src src_SOURCES)

add_executable(RAFT_Libtorch RAFT/main.cpp ${src_SOURCES})
target_link_libraries(RAFT_Libtorch ${OpenCV_LIBRARIES}  ${TORCH_LIBRARIES}
        TorchVision::TorchVision )

