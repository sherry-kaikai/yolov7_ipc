
sail.ipc https://gerrit-ai.sophgo.vip:8443/#/c/93610/


# 例程与测试记录 

## 2023年12月12 patch21
0_resnet_opencv_multi_process.py  success

- 一路进程用cv读取一路视频流，将numpy传入一个长度为args.queue_size的阻塞队列，随后只用一路推理从此queue中取出数据。
- 多路进程从多路视频流读取，多路推理。 
- 此demo只作为参考，不涉及帧的重排等问题。



1_resnet_sail_decode_get_data_from_oneIPC.py error
- 逻辑：多个视频流用一个解码进程，**只用一个ipc管道**传输多个视频流的数据，多个处理进程从此一个管道读取；暂未加结束信号
- 现象(pdb和直接运行均error)：ipc initial success 后报错terminate called after throwing an instance of 'std::system_error'

2_resnet_sail_decode_get_data_from_mulIPC.py success
- 逻辑：每一个视频流都有一个解码进程、一个单独的ipc管道和一个处理进程。
- 现象：pdb run receive后报错terminate called after throwing an instance of 'std::system_error' ，直接运行程序会报同样的错误。也就是ipc读不到图片？
- 174行，如果decode读不到，sleep并不会让程序work
- **推测，ipc还未写入数据，就要读取则报错**
- 解码图片：pdb能运行但是报错 [bmlib_memory][error] free gmem failed!，直接python运行解码图片，卡住。


3_resnet_multi_oneIPC_test.py
- 逻辑：开了一个进程进行multidecoder，开了多个线程进行推理
- 现象：**test1**见第171行，在ipc send处报错；**test2**见191行，卡住。
- 推测：可能是mutidecoder被放入进程的问题？
    - 测试mutidecoder被放入进程能否成功读取
    - **31_multi_test.py** 测试发现
        - test1 error，直接运行，第一次能成功解码完成所有通道，第二次则后报错。pdb运行，第二次解码会直接报错退出
        - test2 success
        - test1和2逻辑不同：就是test1 是多路视频，轮流解码；test2 是多路视频，先解码一路完成再解码一路
4_

## 2023年12月12 patch22
1_resnet_sail_decode_get_data_from_oneIPC.py error
- 逻辑：多个视频流用一个解码进程，**只用一个ipc管道**传输多个视频流的数据，多个处理进程从此一个管道读取；暂未加结束信号
- 现象(pdb和直接运行均error)：ipc initial success 后报错terminate called after throwing an instance of 'std::system_error'

2_resnet_sail_decode_get_data_from_mulIPC.py success
- 逻辑：每一个视频流都有一个解码进程、一个单独的ipc管道和一个处理进程。
- 现象：pdb run receive后报错terminate called after throwing an instance of 'std::system_error' ，直接运行程序会报同样的错误。也就是ipc读不到图片？
- 174行，如果decode读不到，sleep并不会让程序work
- **推测，ipc还未写入数据，就要读取则报错**
- 解码图片：pdb能运行但是报错 [bmlib_memory][error] free gmem failed!，直接python运行解码图片，卡住。


3_resnet_multi_oneIPC_test.py
- 逻辑：开了一个进程进行multidecoder，开了多个线程进行推理
- 现象：pdb和直接运行**test1**解码成功两次，推理成功1次，随后报错，如下；**test2**见191行，**success**。

```
***************************************decode done****************************
[(468, 0.2628248929977417)]
# File[/workingspace/sophon-sail/src/decoder_multi.cpp:324], Thread[139778234910528], channle:[1],Wait: 14.48ms.
***************************************todo decode**************************** 1
***************************************decode done****************************
[2023-12-12 17:06:39.203] [info] [decoder_multi.cpp:471] Start delete channel 0
[2023-12-12 17:06:39.203] [info] [decoder_multi.cpp:278] Channel-0,Set Stop Flag!
[2023-12-12 17:06:39.203] [info] [decoder_multi.cpp:280] Channel-0,Wait Thread Finshed: true!
[(468, 0.2628248929977417)]
[2023-12-12 17:06:39.238] [info] [decoder_multi.cpp:411] <<<<<<<<<<<<<<<<<<<<<<<<<<Channel-0,Decoder Thread Finshed!
```

- 推测：可能是mutidecoder被放入进程的问题？
    - 测试mutidecoder被放入进程能否成功读取
    - **31_multi_test.py** 测试发现
        - test1 error，直接运行，第一次能成功解码完成所有通道，第二次则后报错。pdb运行，第二次解码会直接报错退出
        - test2 success
        - test1和2逻辑不同：就是test1 是多路视频，轮流解码；test2 是多路视频，先解码一路完成再解码一路
### 推测
- ipc在程序异常退出后没有释放？每次运行程序后需要pill python一下
- **ipc还未写入数据，另外的进程就要读取则报错**

# ipc已知缺陷
    
- 目前sail.IPC对象不能作为多进程对象的参数传入multiprocessing.Process
- **sail.IPC无法传输从图片目录中解码得到的图片数据，报错[bmlib_memory][error] free gmem failed!**
- usec2c=true时TPU占用率明显小于usec2c=false,建议使用usec2c=true