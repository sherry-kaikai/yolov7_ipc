本文件夹中的代码不用ipc进行数据传递。
16路：单独开4个进程，每个进程处理4batch，multidecoder add 4channel。进程里面用线程push。

把engine_image_pre_process.PushImage等push和get分别放入线程，然后另一个线程get 。这样会不会更快？ 

传入参数batch_size：模型batch，用于计算跑几个进程。
传入参数video_nums：视频路数。process-nums = video_nums除以模型的batchsize
传入参数loops：每一路视频跑多少张图片后停止。比如传入video_nums=16，loop = 100，总共跑video_nums* loops = 16* 100 = 1600张图片。


- 测试数据 
    - SE5 车牌模型（单类识别） 4batch int8，16路视频，loops=1000。fps=147.8,tpu=100%.

    
增加lprent

- loops for one process is 1000,total fps is 175.2710897575023
- loops for one process is 100,total fps is 89


再次测试

- video nums 24 报错 bm_malloc_device_byte_heap: 661  TPU看内存用了7G

- video nums 16  FPS290?! （把中间一个等待删掉了）
```
INFO:root:Process 2:Loops1000,Total time use: 12033.09178352356 ms, avg_time11.913952260914416, 83.9352028697191 FPS
INFO:root:video nums16, process is 4,total time is 13.749923467636108,loops for one process is 1000,total fps is 290.91071011522376
```

**（相当于 13.6ms 每4batch）**

但是16路存图的时候内存泄漏的问题还是没解决，4路存图内存不泄露但是不能正常退出。 dmesg 看到报错：[1] :ion_ioctl ion alloc failed, fd=-12, from python3
不存图，video = 4 和16可以正常退出

