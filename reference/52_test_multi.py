import argparse
from ctypes import resize
from fileinput import filename
import os
import sys
import time

import cv2
import numpy as np

import multiprocessing

import sophon.sail as sail

import logging
logging.basicConfig(level=logging.INFO)


'''
sail.multi解码视频流
'''
def multidecoder(input_paths, proesss_nums, frame_flag, image_pipe_decode2engine, dist_pipe_decode2engine, tpu_id = 0, max_que_size = 8):
    channel_list = {}
    multiDecoder = sail.MultiDecoder(max_que_size, tpu_id)
    multiDecoder.set_local_flag(True)
    multiDecoder.set_read_timeout(1) # 设置超时时间1s

    print("init 1")
    # ipc = sail.IPC(True, "/tmp/img",  "/tmp/final", usec2c=True)
    print(image_pipe_decode2engine, dist_pipe_decode2engine)
    ipc = sail.IPC(True, image_pipe_decode2engine, dist_pipe_decode2engine, usec2c=True) # 测试一个ipc的情况
    print('init success')
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
    while True:
        for key in channel_list:
            bmimg = sail.BMImage()
            print("***************************************todo decode****************************",int(key))
            ret = multiDecoder.read(int(key),bmimg,read_mode = 1) # 一次处理一个通道 ipc send会报错error
            
            print("retis",ret)
            if ret == 0:
                print("***************************************decode done****************************")
                # ipc.sendBMImage(bmimg, key, frame_id)
                frame_id +=1

            else: 
                time.sleep(0.001)
                # frame_count[int(key)] = 1 # 一个通道数据读完


'''
sail.EngineImagePreProcess 预处理+推理
'''
class Resnet_Engine(object):
    def __init__(self, args):
        self.resize_type =  sail.sail_resize_type.BM_RESIZE_TPU_NEAREST
        self.queue_in_size = 10
        self.queue_out_size = 10

        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.a = [1/(255.*x) for x in self.std]
        self.b = [-x/y for x,y in zip(self.mean, self.std)]
        self.ab = []
        for i in range(3):
            self.ab.append((self.a[i],self.b[i]))


        # init engine 
        # 带有预处理功能的图像推理接口，内部使用线程池的方式，Python下面有更高的效率
        self.engine_image_pre_infer_process = sail.EngineImagePreProcess(args.bmodel, args.dev_id, 0) 
        self.engine_image_pre_infer_process.InitImagePreProcess(self.resize_type, True, self.queue_in_size, self.queue_out_size)
        # self.engine_image_pre_infer_process.SetPaddingAtrr()
        self.engine_image_pre_infer_process.SetConvertAtrr(self.ab)
        self.net_w = self.engine_image_pre_infer_process.get_input_width()
        self.net_h = self.engine_image_pre_infer_process.get_input_height()
        self.output_name = self.engine_image_pre_infer_process.get_output_names()[0]
        self.batch_size = self.engine_image_pre_infer_process.get_output_shape(self.output_name)[0]

        print("init finished")


    '''
    要放在class里面吗，然后在class外面做进程调用？
    放在class里面的好处是能获取到一些class的数据
    '''
    def engine_process(self, image_pipe_decode2engine, dist_pipe_decode2engine, image_pipe_engine2post, dist_pipe_engine2post):
        ipc_decode2engine = sail.IPC(False, image_pipe_decode2engine, dist_pipe_decode2engine, True, 10) # 初始化ipc,接受来自decode的数据
        ipc_engine2post = sail.IPC(True, image_pipe_engine2post, dist_pipe_engine2post, usec2c=True) # 初始化ipc，给出推理后的数据
        while True:
            bmimg, channel_id, frame_id = ipc_decode2engine.receiveBMImage()
            self.engine_image_pre_infer_process.PushImage(channel_id, frame_id, bmimg) # 通道号、帧号传入预处理和推理接口

            '''
            然后直接得到batch data放入ipc？还是说通过传递self.engine_image_pre_infer_process这个实例在后处理来获取呢

            在后处理很快的时候，无所谓，限制已经在推理了
            后处理很慢的时候，还是要传入ipc的
            '''
            output_tensor_map, ost_images, channel_list ,imageidx_list, padding_atrr = self.engine_image_pre_infer_process.GetBatchData(True) # true:搬运数据到系统内存
            tensor_with_name = output_tensor_map[0] # 0指的是第一个batch，如果是4b，长度是4？
            tensor = tensor_with_name.get_data() # 期望从sail.TensorPTRWithName获取到tensor
            ipc_engine2post.sendTensor(tensor, channel_id, frame_id) # 传递tensor

    def postprocess_(output):
        res = list()
        outputs_exp = np.exp(outputs)
        outputs = outputs_exp / np.sum(outputs_exp, axis=1)[:,None]
        predictions = np.argmax(outputs, axis = 1)
        for pred, output in zip(predictions, outputs):
            score = output[pred]
            res.append((pred.tolist(),float(score)))
        return res

    '''
    要放class里面吗，好处还是能获取到一些网络数据
    '''
    def postprocess(self, image_pipe_engine2post, dist_pipe_engine2post):
        ipc_engine2post = sail.IPC(False, image_pipe_engine2post, dist_pipe_engine2post, usec2c=True) 
        # postprocess data
        while True:
            output_tensor, channel_id, frame_id = ipc_engine2post.receiveTensor()
            output = output_tensor.asnumpy() 
            res = self.postprocess_(output)
            print(res)


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    # parser.add_argument('--input', type=str, default='data/images/face_data_test', help='path of input, must be image directory')
    parser.add_argument('--input', type=str, default='../data/monitor.mp4', help='path of input, must be image directory')
    parser.add_argument('--bmodel', type=str, default='../data/BM1684/resnet50_int8_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args




def main(args):
    resnet_engine = Resnet_Engine(args)
    proesss_nums = 2
    input_video = [args.input for _ in range(proesss_nums)]
    frame_flag= multiprocessing.Array("i",[0 for _ in range(proesss_nums)])

    '''
    只使用一个ipc管道
    '''
    image_pipe_decode2engine = "/tmp/img_decode2engine"
    dist_pipe_decode2engine = "/tmp/final_decode2engine"
    image_pipe_engine2post  =  "/tmp/img_engine2post"
    dist_pipe_engine2post = "/tmp/final_engine2post"
    '''
    0 把multidecoder扔到进程里面
    '''
    send_process = multiprocessing.Process(target=multidecoder, args=(input_video, proesss_nums, frame_flag, image_pipe_decode2engine, dist_pipe_decode2engine, 0, 8)) 
    # '''
    # 1 把 sail.EngineImagePreProcess 扔到进程里面
    # '''
    # decode2engine_processes = [multiprocessing.Process(target=resnet_engine.engine_process, args=(image_pipe_decode2engine, dist_pipe_decode2engine, image_pipe_engine2post, dist_pipe_engine2post)) for i in range(proesss_nums)]

    # '''
    # 2 把后处理扔到进程里面
    # '''
    # post_processes = [multiprocessing.Process(target=resnet_engine.postprocess, args=(image_pipe_engine2post, dist_pipe_engine2post)) for i in range(proesss_nums)]
    # # 记录开始时间
    start_time = time.time()

    '''
    start processes
    '''
    # 启动 multidecode 解码
    send_process.start()

    # # 启动推理进程
    # for decode2engine_process in decode2engine_processes:
    #     decode2engine_process.start()

    # # 启动后处理进程
    # for post_process in post_processes:
    #     post_process.start()
    
    
    # 等待进程结束
    send_process.join()

    # for decode2engine_process in decode2engine_processes:
    #     decode2engine_process.join()

    # for post_process in post_processes:
    #     post_process.join()
    
    # 计算时间
    end_time = time.time()
    communication_time = end_time - start_time
    # 打印时间
    print("通信时间：", communication_time, "秒")

if __name__ == '__main__':
    args = argsparser()
    main(args)