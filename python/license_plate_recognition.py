import sophon.sail as sail
import numpy as np

import time
import os
import json
import queue
import argparse
from multiprocessing import Process
import logging
import multiprocessing
import sys
import traceback

from yolov5 import yolov5_process
'''
sail.multi解码视频流
'''
def multidecoder(input_paths, proesss_nums, stop_signal, image_pipe_decode2engine, dist_pipe_decode2engine, tpu_id = 0, max_que_size = 32):

    channel_list = {}

    multiDecoder = sail.MultiDecoder(max_que_size, tpu_id)
    multiDecoder.set_local_flag(True)
    multiDecoder.set_read_timeout(1) # 设置超时时间1s

    ipc = sail.IPC(True, image_pipe_decode2engine, dist_pipe_decode2engine, usec2c=True) 
    logging.info('multidecoder ipc init success')

    if isinstance(input_paths, list):# 多路视频，str地址放在一个list
        for input_path in input_paths:
            channel_index = multiDecoder.add_channel(input_path) # 不丢帧
            logging.info("Add Channel[{}]: {}".format(channel_index, input_path))
            channel_list[channel_index] = input_path


    frame_id = 0
    while True:
        for key in channel_list:
            bmimg = sail.BMImage()
            ret = multiDecoder.read(int(key),bmimg,read_mode = 1) 

            if ret == 0:
                ipc.sendBMImage(bmimg, key, frame_id)
                frame_id +=1
                logging.info("decode channel and sent to ipc done, channle id is %d, frameid is %d",key,frame_id)
                print(frame_id)            
            else: 
                time.sleep(0.001)
        if frame_id == len(input_paths)*100:
            stop_signal.value = True
            break
    logging.info('decode done')
    try:
        sys.exit(0)  # 正常退出，退出状态码为0
    except SystemExit as e:
        exit_code = e.code
        print("multidecode Exit code:", exit_code)

import signal
def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        # 获取当前进程的子进程 ID
        child_pids = os.popen(f"pgrep -P {parent_pid}").read().splitlines()
    except Exception as e:
        print(f"无法获取子进程的 ID：{e}")
        return

    # 终止子进程
    for pid in child_pids:
        try:
            os.kill(int(pid), sig)
            print(f"已终止子进程：{pid}")
        except Exception as e:
            print(f"无法终止子进程 {pid}：{e}")

def main(args):
    try:
        proesss_nums = args.proesss_nums
    
        '''
        只使用一个ipc管道
        '''
        image_pipe_decode2engine = "/tmp/img"
        dist_pipe_decode2engine = "/tmp/final"
        image_pipe_engine2post  = "/tmp/img_engine2post"
        dist_pipe_engine2post = "/tmp/final_engine2post"


        # 创建一个共享的布尔变量作为结束信号
        stop_signal = multiprocessing.Value('b', False)

        '''
        4解码进程，一个解码器解码4路
        '''
        # input_video = [args.input for _ in range(proesss_nums)]
        # send_process = [multiprocessing.Process(target=multidecoder, args=(input_video, proesss_nums, stop_signal, image_pipe_decode2engine, dist_pipe_decode2engine, args.dev_id, 8)) for i in range(proesss_nums)]
        
        '''
        单进程，解码器解码16路
        '''
        input_video = [args.input for _ in range(proesss_nums*4)]
        send_process = multiprocessing.Process(target=multidecoder, args=(input_video, proesss_nums, stop_signal, image_pipe_decode2engine, dist_pipe_decode2engine, args.dev_id, args.multidecode_max_que_size)) 
        '''
        4进程，每个进程4batch
        '''
        receive = [multiprocessing.Process(target=yolov5_process,args=(stop_signal, args.ipc_recive_queue_len, args.yolo_bmodel, image_pipe_decode2engine, dist_pipe_decode2engine,args.dev_id)) for i in range(proesss_nums)]
        start_time = time.time()
        logging.info(start_time)

        send_process.start()
        for i in receive:
            i.start()
        logging.debug('start decode and yolo procrss')
        try:
            send_process.join()
            for i in receive:
                i.join()
            logging.debug('DONE decode and yolo process')
        except:
            sys.exit(1)

        total_time = time.time() - start_time
        logging.info('total time {}'.format(total_time))
        logging.info('total fps %.2f'% (total_time/1000))

        try:
            # os.kill(os.getpid(),9)
            parent_pid = os.getpid()
            # 终止当前程序运行带来的所有子进程
            kill_child_processes(parent_pid)
        except:
            pass

    
    except Exception as e:
        # 捕获异常
        print("An exception occurred:", e)
        # 打印异常堆栈信息
        traceback.print_exc()
        # 可以选择退出程序
        sys.exit(1)



def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--multidecode_max_que_size', type=int, default=32, help='multidecode queue')
    parser.add_argument('--ipc_recive_queue_len', type=int, default=16, help='ipc recive queue')
    parser.add_argument('--chip_mode', type=str, default='1684x', help='1684x or 1684')
    parser.add_argument('--proesss_nums', type=int, default=4, help='procress nums of process and postprocess')
    parser.add_argument('--input', type=str, default='../data/licenseplate.mp4', help='path of input, must be video path') 
    parser.add_argument('--yolo_bmodel', type=str, default='../models/yolov5s-licensePLate/BM1684X/yolov5s_v6.1_license_3output_int8_4b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()

    name_number = args.proesss_nums
    chip_mode = args.chip_mode
    if chip_mode == '1684':
        args.yolo_bmodel = '../models/yolov5s-licensePLate/BM1684/yolov5s_v6.1_license_3output_int8_4b.bmodel'
    logging.basicConfig(filename= f'{chip_mode}_process_is_{name_number}.log',filemode='w',level=logging.DEBUG)
    try:
        main(args)
    except Exception as e:
        print("An exception occurred:", e)
        # 打印异常堆栈信息
        traceback.print_exc()
        # 可以选择退出程序
        sys.exit(1)

