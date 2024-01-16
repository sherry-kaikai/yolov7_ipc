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