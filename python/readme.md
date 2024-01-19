本文件夹中的代码不用ipc进行数据传递。
16路：单独开4个进程，每个进程处理4batch，multidecoder add 4channel。

需要把engine_image_pre_process.PushImage 放入一个线程吗，然后另一个线程get 。这样会不会更快？ 当前文件夹没做这个。

传入参数batch_size：模型batch，用于计算跑几个进程。
传入参数video_nums：视频路数。process-nums = video_nums除以模型的batchsize
传入参数loops：每一路视频跑多少张图片后停止。比如传入video_nums=16，loop = 100，总共跑video_nums* loops = 16* 100 = 1600张图片。


- 测试数据 
    - SE5 车牌模型（单类识别） 4batch int8，16路视频，loops=100。fps=126,tpu=100%.
    - SE5 COCO（80类） 4batch int8,16路视频，nms=0.6，dete=0.01，fps=116. dete=0.65，fps=117.
    - SE7 车牌（单类识别） 4batch int8 16路视频。dete 0.65 nms 0.65,  fps250  tpu 80~90% . 
    - SE7 coco（80类）4batch int8 16路视频。dete 0.65 nms 0.65,  fps140  tpu 80~90% . dete 0.01 nms 0.65 fps fps95  tpu 70%


