## 2024年1月16日
1 进程解码
1 进程推理（只有yolo）

最后退出的时候会'std::system_error'是ipc的问题

## 2024年1月16日 
1 进程解码 16路
4 进程推理（只有yolo）

1684 SE5 2s跑160张图

### 15点48分 
- pcie 1684x  1600张图 int8 4b   

    tpu_util = 60~80%
    1705391315.8426158 - 1705391310.45986 = 5.84 - 0.46 = 5.4s
    1600/5.4 = 296fps

    (test2:9.8684905 - 4.5216286 = 5.4s 

    INFO:root:YOLO pre and process done, time use: 0.05s
    INFO:root:YOLO postprocess init done, time use: 0.00s
    INFO:root:YOLO postprocess done, time use: 0.00s
    INFO:root:YOLO crop done, time use: 0.02s

    INFO:root:YOLO pre and process done, time use: 0.06s
    INFO:root:YOLO postprocess init done, time use: 0.00s
    INFO:root:YOLO postprocess done, time use: 0.00s
    YOLO crop done, time use: 0.01s
    ----
    parser.add_argument('--multidecode_max_que_size', type=int, default=32, help='multidecode queue')
    parser.add_argument('--ipc_recive_queue_len', type=int, default=16, help='ipc recive queue')
    10.2887964 - 4.8115504 = 5.4
    1600/5.4 = 

- soc 1684
    - test1   
    parser.add_argument('--multidecode_max_que_size', type=int, default=32, help='multidecode queue')
    parser.add_argument('--ipc_recive_queue_len', type=int, default=16, help='ipc recive queue')
    把视频换成了VPU能解码的h264 64对齐的，2987MB/ 9135MB ,tpu不到100，差不多最高96到98左右
    1600张图解码，但是应该只crop到1532张图，就按照1600算吧
    87.4545023 - 76.0292306 = 11.42
    1600/11.42 = 140 fps
    一路8.75fps 

    - test2
    parser.add_argument('--multidecode_max_que_size', type=int, default=16, help='multidecode queue')
    parser.add_argument('--ipc_recive_queue_len', type=int, default=16, help='ipc recive queue')

    2755MB/ 9135MB  tpu~100

    67.8697917 - 56.3816276 == 11.48

    - test3 8,16 2642MB/ 9135MB  ~100
    17.672545 - 06.2309382 = 11.44

    - test4 16,16   2782MB/ 9135MB
    331.422377 - 319.9720676 = 11.450

    - 记录的真实数据，预处理和推理要比预期慢一些：
    
    ```
    INFO:root:pushdata exit, time use: 0.00s
    YOLO pre and process done, time use: 0.05s
    INFO:root:YOLO postprocess init done, time use: 0.00s
    INFO:root:YOLO postprocess done, time use: 0.00s
    INFO:root:YOLO crop done, time use: 0.01s

    INFO:root:YOLO pre and process done, time use: 0.09s
    # 中间步骤都是0.00s
    INFO:root:YOLO crop done, time use: 0.01s

    ```
    理论：50*4 fps = 1000/200  每路5fps，16路

    为啥get 预处理+推理的速度那么慢 ，demo的数据是：
    | BM1684 SoC  | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 2.8      | 2.3           | 11.5          | 115       |
    0.021574


    **可能是拷贝用时比较多 need_d2s 后处理要用**


## 2024年1月18日
### 测试engineimagepreprocess d2s的时间 16路  decode队列长32，ipc队列长16（理论上这两保持一致就行）

4进程：
- 不做d2s：
    总时间
    INFO:root:YOLO pre and process done,without d2s, time use: 0.04s
    INFO:root:YOLO pre and process done,without d2s, time use: 0.07s
    INFO:root:YOLO pre and process done,without d2s, time use: 0.08s
    INFO:root:YOLO pre and process done,without d2s, time use: 0.09s
    INFO:root:YOLO pre and process done,without d2s, time use: 0.13s
    INFO:root:YOLO pre and process done,without d2s, time use: 0.11s
    INFO:root:YOLO pre and process done,without d2s, time use: 0.09s

    细节时间：
    INFO:root:pushdata exit, time use: 0.0018s
    INFO:root:pushdata exit, time use: 0.0016s
    INFO:root:pushdata exit, time use: 0.0015s
    INFO:root:pushdata exit, time use: 0.0017s
    INFO:root:YOLO pre and process done,without d2s, get one batch data time use: 0.0442s
    INFO:root:YOLO pre and process done,without d2s, total time use: 0.0467s

    INFO:root:YOLO pre and process done,without d2s, get one batch data time use: 0.0700s
    INFO:root:YOLO pre and process done,without d2s, total time use: 0.0721s

    INFO:root:YOLO pre and process done,without d2s, get one batch data time use: 0.0937s
    INFO:root:YOLO pre and process done,without d2s, total time use: 0.0958s

- 做d2s：（后面时间越来越长可能是数据没出队？ 4进程查可能看不出来）
    INFO:root:YOLO pre and process done,with d2s, get one batch data time use: 0.0488s
    INFO:root:YOLO pre and process done,with d2s, total time use: 0.0509s

    INFO:root:YOLO pre and process done,with d2s, get one batch data time use: 0.0731s
    INFO:root:YOLO pre and process done,with d2s, total time use: 0.0747s

    INFO:root:YOLO pre and process done,with d2s, get one batch data time use: 0.0927s
    INFO:root:YOLO pre and process done,with d2s, total time use: 0.0951s

    INFO:root:YOLO pre and process done,with d2s, get one batch data time use: 0.0957s
    INFO:root:YOLO pre and process done,with d2s, total time use: 0.0978s   

1进程 4channal  multidecode,ipc 都是16长
400 张图 56.56387-51.613986 = 5s  400/5 = 80fps

- 不做d2s：
    INFO:root:YOLO pre and process done,with out d2s, get one batch data time use: 0.0359s
    INFO:root:YOLO pre and process done,with out d2s, total time use: 0.0368s

    INFO:root:YOLO pre and process done,with out d2s, get one batch data time use: 0.0360s
    INFO:root:YOLO pre and process done,with out d2s, total time use: 0.0370s

    INFO:root:YOLO pre and process done,with out d2s, get one batch data time use: 0.0377s
    INFO:root:YOLO pre and process done,with out d2s, total time use: 0.0387s

    INFO:root:YOLO pre and process done,with out d2s, get one batch data time use: 0.0369s
    INFO:root:YOLO pre and process done,with out d2s, total time use: 0.0379s

- 做d2s：
    INFO:root:YOLO pre and process done,with d2s, get one batch data time use: 0.0367s
    INFO:root:YOLO pre and process done,with d2s, total time use: 0.0387s

    INFO:root:YOLO pre and process done,with d2s, get one batch data time use: 0.0367s
    INFO:root:YOLO pre and process done,with d2s, total time use: 0.0377s

    INFO:root:YOLO pre and process done,with d2s, get one batch data time use: 0.0367s
    INFO:root:YOLO pre and process done,with d2s, total time use: 0.0381s

    INFO:root:YOLO pre and process done,with d2s, get one batch data time use: 0.0361s
    INFO:root:YOLO pre and process done,with d2s, total time use: 0.0377s

    INFO:root:YOLO pre and process done,with d2s, get one batch data time use: 0.0362s
    INFO:root:YOLO pre and process done,with d2s, total time use: 0.0376s

### 测试不用ipc 开4个单进程
