# RAFT_Libtorch
这是光流算法RAFT的Libtorch实现。

目前使用两种方式将pytorch模型转换为libtorch模型。  
一种使用torchscript编译的方法，对应代码中`RAFT_TorchScript类`。  
另一种使用trace的方法，对应代码中的`RAFT_Torch类`。由于RAFT中存在分支和循环语句，因此这种方法需要分别
将RAFT中三个子网络分别trace。  
这两种方法所需要的显存均非常大，在我的2070显卡（8G现存）上都会出现
显存不够的情况。

## Quick Start
* **1.编译TorchScript模型**
    这里通过TorchScript编译器将原来的Pytorch模型转换为Libtorch模型，过程请参考我修改的
[RAFT](https://github.com/chenjianqu/RAFT)  

* **2.下载并编译**   
```shell
git clone https://github.com/chenjianqu/RAFT_Libtorch.git

cd RAFT_Libtorch
mkdir build && cd build

cmake ..
make -j10
```
* **3.运行程序**  
首先编辑配置文件`config.yaml`，设置相应的路径。
```yaml
model_path: "../weights/kitti.pt"
segmentor_log_path: "segmentor_log.txt"
segmentor_log_level: "debug"
segmentor_log_flush: "debug"

#DATASET_DIR: "/media/chen/EC4A17F64A17BBF0/datasets/kitti/odometry/colors/07/image_2/"
DATASET_DIR: "/home/chen/CLionProjects/RAFT_CPP/demo/kitti07/"
WARN_UP_IMAGE_PATH: "/home/chen/CLionProjects/InstanceSegment/config/kitti.png"
```
`model_path`,`DATASET_DIR`这两项必须设置正确。
  
运行程序
```shell
./RAFT_Libtorch ../config/config.yaml
```




