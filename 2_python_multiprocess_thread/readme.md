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
