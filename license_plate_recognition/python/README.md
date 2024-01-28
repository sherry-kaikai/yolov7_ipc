[简体中文](./README.md)

# python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 程序编译](#2-程序编译)
    * [2.1 x86/arm PCIe平台](#21-x86arm-pcie平台)
    * [2.2 SoC平台](#22-soc平台)
* [3. 推理测试](#3-推理测试)
    * [3.1 参数说明](#31-参数说明)
    * [3.2 测试图片](#32-测试图片)
    * [3.3 测试视频](#33-测试视频)

python目录下提供了python例程以供参考使用，具体情况如下：
| 序号  | python例程                   | 说明                                 |
| ---- | ----------------------------- | -----------------------------------  |
| 1    | license_plate_recognition.py  | 使用opencv解码、BMCV前处理、BMRT推理 |
| 2    | chars.py                      | lprnet后处理使用的汉字字典           |


## 1. 环境准备
### 1.1 x86/arm PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv、sophon-ffmpeg、sophon-sail，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。
您还需要安装sophon-sail，具体步骤可参考[编译安装sophon-sail](../../../docs/Environment_Install_Guide.md##3.3 编译安装sophon-sail)。

## 3. 推理测试
对于PCIe平台和SoC平台，均可以直接在进行推理测试。

### 3.1 参数说明


|   参数名               | 类型    |   说明                              |
|------------------------|---------|-------------------------------------|
| max_que_size           | int     |   设备号                            |
| video_nums             | string  |   视频测试路数                      |
| batch_size             | int     |   输入bmodel的batch_size            |
| loops                  | int     |   对于一个进程的循环测试图片数      |
| input                  | string  |   本地视频路径或视频流地址          |
| yolo_bmodel            | int     |   yolov5 bmodel路径                 |
| lprnet_bmodel          | int     |   lprnet bmodel路径                 |
| dev_id                 | int     |   推理使用的设备id                  |
| draw_images            | int     |   是否保存图片                      |


### 3.2 运行程序
运行应用程序即可
```bash
python3 license_plate_recognition.py --input /data/licenseplate_640516-h264.mp4  --loops 1000 --video_nums 16
```
测试过程会打印被检测和识别到的有效车牌信息，测试结束后，会在log中打印fps等信息。