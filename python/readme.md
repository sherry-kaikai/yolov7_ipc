本文件夹中的代码不用ipc进行数据传递。
16路：单独开4个进程，每个进程处理4batch，multidecoder add 4channel。

需要把engine_image_pre_process.PushImage 放入一个线程吗，然后另一个线程get 。这样会不会更快？ 当前文件夹没做这个。

传入参数batch_size：模型batch，用于计算跑几个进程。
传入参数video_nums：视频路数。process-nums = video_nums除以模型的batchsize
传入参数loops：每一路视频跑多少张图片后停止。比如传入video_nums=16，loop = 100，总共跑video_nums* loops = 16* 100 = 1600张图片。


- 测试数据 SE5 4batch int8，16路视频，loops=100。fps=126,tpu=100%