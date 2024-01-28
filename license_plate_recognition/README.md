[简体中文](./README.md)

# LPRNet

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
* [5. 例程测试](#5-例程测试)
* [6. 性能测试](#6-性能测试)
  * [6.1 bmrt_test](#61-bmrt_test)
  * [6.2 程序运行性能](#62-程序运行性能)
  
## 1. 简介

本例程用于说明如何使用 python 和 sophon.sail 快速构建基于 yolov5 的车牌检测和基于 lprnet 的车牌识别。并在SOPHON BM1684/BM1684X/BM1688上进行推理测试。

**LPRNET 车牌检测源代码**(https://github.com/sirius-ai/LPRNet_Pytorch)

本例程中，yolov5、lprnet 算法的前处理、推理、后处理均分别在多个线程上进行运算，保证了一定的检测效率。

## 2. 特性
* 支持BM1688(SoC)、BM1684X(x86 PCIe、SoC)、BM1684(x86 PCIe、SoC、arm PCIe)
* LPRNet支持FP32、FP16(BM1684X)、INT8模型编译和推理
* YOLOv5支持FP32、INT8模型编译和推理
* 支持基于sail.EngineImagePreProcess的预处理和推理，基于sail.algo_yolov5_post_cpu_opt_async的yolo后处理加速接口
* 支持多batch模型推理
* 支持图片和视频测试
* pipeline式demo，支持解码、预处理和后处理多线程和推理线程并行运行，更充分利用硬件加速

## 3. 准备模型与数据
​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
chmod -R +x scripts/
./scripts/download.sh

```
执行后，模型保存至`models/`，数据集下载并解压至`datasets/`
```
下载的模型包括：
./models
├── lprnet
|   ├── BM1684
|   │   ├── lprnet_fp32_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684的FP32 LPRNet BModel，batch_size=1，num_core=1
|   │   ├── lprnet_int8_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684的INT8 LPRNet BModel，batch_size=1，num_core=1
|   │   └── lprnet_int8_4b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684的INT8 LPRNet BModel，batch_size=4，num_core=1
|   ├── BM1684X
|   │   ├── lprnet_fp32_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684X的FP32 LPRNet BModel，batch_size=1，num_core=1
|   │   ├── lprnet_fp16_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684X的FP16 LPRNet BModel，batch_size=1，num_core=1
|   │   ├── lprnet_int8_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684X的INT8 LPRNet BModel，batch_size=1，num_core=1
|   │   └── lprnet_int8_4b.bmodel                                     # 使用TPU-MLIR编译，用于BM1684X的INT8 LPRNet BModel，batch_size=4，num_core=1
|   ├── BM1688
|   │   ├── lprnet_fp32_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1688的FP32 LPRNet BModel，batch_size=1，num_core=1
|   │   ├── lprnet_fp16_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1688的FP16 LPRNet BModel，batch_size=1，num_core=1
|   │   ├── lprnet_int8_1b.bmodel                                     # 使用TPU-MLIR编译，用于BM1688的INT8 LPRNet BModel，batch_size=1，num_core=1
|   │   ├── lprnet_int8_4b.bmodel                                     # 使用TPU-MLIR编译，用于BM1688的INT8 LPRNet BModel，batch_size=4，num_core=1
|   │   ├── lprnet_fp32_1b_2core.bmodel                               # 使用TPU-MLIR编译，用于BM1688的FP32 LPRNet BModel，batch_size=1，num_core=2
|   │   ├── lprnet_fp16_1b_2core.bmodel                               # 使用TPU-MLIR编译，用于BM1688的FP16 LPRNet BModel，batch_size=1，num_core=2
|   │   ├── lprnet_int8_1b_2core.bmodel                               # 使用TPU-MLIR编译，用于BM1688的INT8 LPRNet BModel，batch_size=1，num_core=2
|   │   └── lprnet_int8_4b_2core.bmodel                               # 使用TPU-MLIR编译，用于BM1688的INT8 LPRNet BModel，batch_size=4，num_core=2
|   │── torch
|   │   ├── Final_LPRNet_model.pth                                    # LPRNet 原始模型
|   │   └── LPRNet_model_trace.pt                                     # trace后的JIT LPRNet模型
|   └── onnx
|       ├── lprnet_1b.onnx                                            # 导出的onnx LPRNet模型，batch_size=1
|       └── lprnet_4b.onnx                                            # 导出的onnx LPRNet模型，batch_size=4   
└── yolov5s-licensePLate
    ├── BM1684
    │   ├── yolov5s_v6.1_license_3output_fp32_1b.bmodel               # 用于BM1684的FP32 YOLOv5 BModel，batch_size=1，num_core=1
    |   └── yolov5s_v6.1_license_3output_int8_1b.bmodel               # 用于BM1684的INT8 YOLOv5 BModel，batch_size=1，num_core=1
    ├── BM1684X
    │   ├── yolov5s_v6.1_license_3output_fp32_1b.bmodel               # 用于BM1684X的FP32 YOLOv5 BModel，batch_size=1，num_core=1
    │   └── yolov5s_v6.1_license_3output_int8_1b.bmodel               # 用于BM1684X的INT8 YOLOv5 BModel，batch_size=1，num_core=1
    └── BM1688
        ├── yolov5s_v6.1_license_3output_fp32_1b_2core.bmodel         # 用于BM1688的FP32 YOLOv5 BModel，batch_size=1，num_core=2
        ├── yolov5s_v6.1_license_3output_fp32_1b.bmodel               # 用于BM1688的FP32 YOLOv5 BModel，batch_size=1，num_core=1
        ├── yolov5s_v6.1_license_3output_fp32_4b_2core.bmodel         # 用于BM1688的FP32 YOLOv5 BModel，batch_size=4，num_core=2
        ├── yolov5s_v6.1_license_3output_fp32_4b.bmodel               # 用于BM1688的FP32 YOLOv5 BModel，batch_size=4，num_core=1
        ├── yolov5s_v6.1_license_3output_int8_1b_2core.bmodel         # 用于BM1688的INT8 YOLOv5 BModel，batch_size=1，num_core=2
        ├── yolov5s_v6.1_license_3output_int8_1b.bmodel               # 用于BM1688的INT8 YOLOv5 BModel，batch_size=1，num_core=1
        ├── yolov5s_v6.1_license_3output_int8_4b_2core.bmodel         # 用于BM1688的INT8 YOLOv5 BModel，batch_size=4，num_core=2
        └── yolov5s_v6.1_license_3output_int8_4b.bmodel               # 用于BM1688的INT8 YOLOv5 BModel，batch_size=4，num_core=1

下载的数据包括：
./datasets
├── licenseplate_640516-h264.mp4            # 测试视频1
└── 1080_1920_5s.mp4                        # 测试视频2

```

## 4. 例程测试
- [python例程](./python/README.md)

## 5. 性能测试
### 5.1 bmrt_test
可参考[LPRNet bmrt_test](../../sample/LPRNet/README.md#71-bmrt_test)里面的性能数据。
可参考[YOLOv5 bmrt_test](../../sample/YOLOv5/README.md#71-bmrt_test)里面的性能数据。

### 5.2 程序运行性能
参考[python例程](./python/README.md)运行程序，并查看统计的total fps。

在不同的测试平台上，测试视频`datasets/licenseplate_640516-h264.mp4`，模型`lprnet_int8_4b.bmodel，yolov5s_v6.1_license_3output_int8_4b.bmodel`，`--loops=1000` ，`--draw_images=False`，性能测试结果如下：
|         测试平台         |      测试程序                 |             测试模型                                              | 视频路数 | fps  | tpu-util | cpu-util（单核）   | 系统内存占用 | 设备内存占用  |
| ------------------------ | ----------------------------- | ----------------------------------------------------------------- | --------| ---- | -------- | ------------------ | ----------- | ------------- |
| BM1684 SoC, Currclk 550M | license_plate_recognition.py  | lprnet_int8_4b.bmodel，yolov5s_v6.1_license_3output_int8_4b.bmodel|   16    |  291 |  100%    |      30% 左右      |   1.9%左右  |    2500MB左右 |
| BM1684 SoC, Currclk 550M | license_plate_recognition.py  | lprnet_int8_4b.bmodel，yolov5s_v6.1_license_3output_int8_4b.bmodel|   24    |  388 |  100%    |      30% 左右      |   1.9%左右  |    3700MB左右 |
| BM1684 SoC, Currclk 550M | license_plate_recognition.py  | lprnet_int8_4b.bmodel，yolov5s_v6.1_license_3output_int8_4b.bmodel|   32    |  447 |  100%    |      30% 左右      |   1.9%左右  |    4700MB左右 |
| BM1684 SoC, Currclk 330M | license_plate_recognition.py  | lprnet_int8_4b.bmodel，yolov5s_v6.1_license_3output_int8_4b.bmodel|   16    |  257 |  100%    |      30% 左右      |   2.6%左右  |    2500MB左右 |
| BM1684 SoC, Currclk 330M | license_plate_recognition.py  | lprnet_int8_4b.bmodel，yolov5s_v6.1_license_3output_int8_4b.bmodel|   24    |  324 |  100%    |      30% 左右      |   2.6%左右  |    3700MB左右 |
| BM1684 SoC, Currclk 330M | license_plate_recognition.py  | lprnet_int8_4b.bmodel，yolov5s_v6.1_license_3output_int8_4b.bmodel|   32    |  479 |  100%    |      30% 左右      |   2.6%左右  |    3000MB~3800MB |
| BM1684X SoC, Currclk 1000M| license_plate_recognition.py | lprnet_int8_4b.bmodel，yolov5s_v6.1_license_3output_int8_4b.bmodel|   16    |  350 |  100%    |      70%左右     |   1.0%左右  |    2500MB左右  |
| BM1684X SoC, Currclk 1000M| license_plate_recognition.py | lprnet_int8_4b.bmodel，yolov5s_v6.1_license_3output_int8_4b.bmodel|   24    |  508 |  100%    |      70%左右     |   1.0%左右  |    3700MB左右  |
| BM1684X SoC, Currclk 1000M| license_plate_recognition.py | lprnet_int8_4b.bmodel，yolov5s_v6.1_license_3output_int8_4b.bmodel|   32    |  628 |  100%    |      70%左右     |   1.0%左右  |    5100MB左右  |


> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. BM1684/1684X SoC的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异； 




