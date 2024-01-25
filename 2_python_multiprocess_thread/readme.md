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

去掉lprnet

- 只测试yolo 32路和16路视频都是144fps ..奇怪 TPU稳定100% queue长度32（所以和queue长度无关）

```
INFO:root:Process 2:Loops2000,Total time use: 54779.06131744385 ms, avg_time27.389530658721924, 36.51030068606013 FPS
INFO:root:Engine_image_pre_process GetBatchData time use: 82.18 ms
DEBUG:root:LOOPS DONE
INFO:root:Process 3:Loops2000,Total time use: 54774.441957473755 ms, avg_time27.387220978736877, 36.51337975387822 FPS
INFO:root:Engine_image_pre_process GetBatchData time use: 83.91 ms
INFO:root:Engine_image_pre_process GetBatchData time use: 85.56 ms
DEBUG:root:DONE decode and yolo process
DEBUG:root:DONE decode and yolo process
DEBUG:root:DONE decode and yolo process
DEBUG:root:DONE decode and yolo process
INFO:root:video nums16, process is 4,total time is 55.309465169906616,loops for one process is 2000,total fps is 144.64070436090074
```

```
INFO:root:Process 7:Loops2000,Total time use: 109762.37893104553 ms, avg_time54.881189465522766, 18.221179419374938 FPS
INFO:root:Engine_image_pre_process GetBatchData time use: 81.82 ms
DEBUG:root:DONE decode and yolo process
DEBUG:root:DONE decode and yolo process
DEBUG:root:DONE decode and yolo process
DEBUG:root:DONE decode and yolo process
DEBUG:root:DONE decode and yolo process
INFO:root:video nums32, process is 8,total time is 110.4132616519928,loops for one process is 2000,total fps is 144.9101290968993

```