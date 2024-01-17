import argparse
from ctypes import resize
from fileinput import filename
import os
import sys
import time

import numpy as np

import multiprocessing

import sophon.sail as sail

def multidecoder(input_paths, proesss_nums, frame_count, image_pipe:str,dist_pipe:str,tpu_id = 0, max_que_size = 32):
    channel_list = {}
    multiDecoder = sail.MultiDecoder(max_que_size, tpu_id)
    multiDecoder.set_local_flag(True)
    multiDecoder.set_read_timeout(1) # 设置超时时间1s
    # ipc = sail.IPC(True, image_pipe, dist_pipe) # 测试一个ipc的情况

    if isinstance(input_paths, list):# 多路视频，str地址放在一个list
        for input_path in input_paths:
            print('************************add channel todo************************')
            channel_index = multiDecoder.add_channel(input_path) # 不丢帧
            print("Add Channel[{}]: {}".format(channel_index, input_path))
            channel_list[channel_index] = input_path
            print('************************added channel************************',input_path)
    elif os.path.isfile(input_paths): # 单视频
        # print('************************is singel video************************')
        # multidecoder
        channel_index = multiDecoder.add_channel(input_paths,1)
        print("Add Channel[{}]: {}".format(channel_index,input_paths))
        channel_list[channel_index] = input_paths
    elif os.path.isdir(input_paths): # 文件夹
        for root, dirs, filenames in os.walk(input_paths):
            for filename in filenames:
                print('************************',filename,'************************')
                img_file = os.path.join(root, filename)
                # TODO

    frame_id = 0
    '''
    test1 error, 第一次能成功解码完成所有通道，第二次则后报错。
    即 decode_times = 0,无法到1 。

    pdb 直接run会decode_times = 0成功解码，到decode_times = 1就报错Channel-0,Set Stop Flag!
    '''
    decode_times = 0
    while True:
        
        print("decode times: ",decode_times)
        
        decode_times += 1
        for key in channel_list:
            
            bmimg = sail.BMImage()
            ret = multiDecoder.read(int(key),bmimg,read_mode = 1) # 一次处理一个通道 ipc send会报错error
            time.sleep(0.1)
            print("***************************************todo decode,channle id is****************************",int(key) )
            if ret == 0:
                print("***************************************decode done****************************")
                # ipc.sendBMImage(bmimg, key, frame_id)
                frame_id +=1

            else: 
                time.sleep(0.001)
                # frame_count[int(key)] = 1 # 一个通道数据读完
                # break
        # break #所有通道数据读完

    '''
    test2  success
    '''
    # for key in channel_list:
    #     while True:
    #         bmimg = sail.BMImage()
    #         ret = multiDecoder.read(int(key),bmimg,read_mode = 1) # 一次处理一个通道 ipc send会报错error
            
    #         print("***************************************todo decode****************************",int(key) )
    #         if ret == 0:
    #             print("***************************************decode success****************************")
    #             # ipc.sendBMImage(bmimg, key, frame_id)
    #             print("***************************************ipc send success****************************")
    #             frame_id +=1

    #         else: 
    #             time.sleep(0.001)
    #             print("***************************************decode error****************************")
    #             # frame_count[int(key)] = 1 # 一个通道数据读完
    #             break


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    # parser.add_argument('--input', type=str, default='data/images/face_data_test', help='path of input, must be image directory')
    # parser.add_argument('--input', type=str, default='/data/monitor.mp4', help='path of input, must be image directory')
    parser.add_argument('--input', type=str, default='/data/licenseplate.mp4', help='path of input, must be image directory')
    parser.add_argument('--bmodel', type=str, default='../data/BM1684X/resnet50_int8_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args

def main(args):

    proesss_nums = 16
    input_video = [args.input for _ in range(proesss_nums)]
    frame_flag= multiprocessing.Array("i",[0 for _ in range(proesss_nums)])

    image_pipe = "/tmp/img"
    dist_pipe = "/tmp/final"
    '''
    把multidecoder扔到进程里面似乎是需要的，否则怎么同时开启进程呢
    '''
    send_process = multiprocessing.Process(target=multidecoder, args=(input_video, proesss_nums, frame_flag, image_pipe, dist_pipe, 0, 32)) 


    '''
    单独multidecoder程序
    '''
    # multidecoder(input_video, proesss_nums, frame_flag, image_pipe, dist_pipe, tpu_id = 0, max_que_size = 8)



    # 启动multidecode 解码
    send_process.start()

    
    
    send_process.join()



if __name__ == '__main__':
    args = argsparser()
    main(args)
