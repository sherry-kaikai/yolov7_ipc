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


class Resnet(object):
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))

        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.input_shapes = [self.net.get_input_shape(self.graph_name, name) for name in self.input_names]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_shapes = [self.net.get_output_shape(self.graph_name, name) for name in self.output_names]
        logging.debug(str(("graph_name: {}, input_names & input_shapes: ".format(self.graph_name), self.input_names, self.input_shapes)))
        logging.debug(str(("graph_name: {}, output_names & output_shapes: ".format(self.graph_name), self.output_names, self.output_shapes)))

        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_name)
        # define input and ouput shape
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        # init bmcv for preprocess
        self.handle = self.net.get_handle()
        self.bmcv = sail.Bmcv(self.handle)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)

        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.output_scale = self.net.get_output_scale(self.graph_name, self.output_name)

        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

        self.a = [1/(255.*x) for x in self.std]
        self.b = [-x/y for x,y in zip(self.mean, self.std)]

        self.ab = []
        for i in range(3):
            self.ab.append(self.a[i]*self.input_scale)
            self.ab.append(self.b[i]*self.input_scale)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

        self.output_tensor = sail.Tensor(self.handle, self.output_shape, self.output_dtype,  True, True)
        self.output_tensors = {self.output_name: self.output_tensor}

    def preprocess_bmcv(self, input_bmimg, output_bmimg):
        if input_bmimg.format()==sail.Format.FORMAT_YUV420P:
            input_bmimg_bgr = self.bmcv.yuv2bgr(input_bmimg)
        else:
            input_bmimg_bgr = input_bmimg

        resize_bmimg = self.bmcv.resize(input_bmimg_bgr, self.net_w, self.net_h, sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)
        resize_bmimg_rgb = sail.BMImage(self.handle, self.net_h, self.net_w, \
                        sail.Format.FORMAT_RGB_PLANAR, resize_bmimg.dtype())
        self.bmcv.convert_format(resize_bmimg, resize_bmimg_rgb)
        self.bmcv.convert_to(resize_bmimg_rgb, output_bmimg, ((self.ab[0], self.ab[1]), \
                                       (self.ab[2], self.ab[3]), \
                                       (self.ab[4], self.ab[5])))
        return output_bmimg

    def predict(self, input_tensor):
        input_tensors = {self.input_name: input_tensor}
        self.net.process(self.graph_name, input_tensors, self.output_tensors)
        outputs = self.output_tensor.asnumpy() * self.output_scale
        return outputs

    def postprocess(self, outputs):
        res = list()
        outputs_exp = np.exp(outputs)
        outputs = outputs_exp / np.sum(outputs_exp, axis=1)[:,None]
        predictions = np.argmax(outputs, axis = 1)
        for pred, output in zip(predictions, outputs):
            score = output[pred]
            res.append((pred.tolist(),float(score)))
        return res

    def __call__(self, bmimg_list):
        img_num = len(bmimg_list)
        assert img_num <= self.batch_size

        if self.batch_size == 1:
            for bmimg in bmimg_list:            
                output_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, \
                        sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
                start_time = time.time()
                output_bmimg = self.preprocess_bmcv(bmimg, output_bmimg)
                self.preprocess_time += time.time() - start_time
                input_tensor = sail.Tensor(self.handle, self.input_shape,  self.input_dtype,  False, False)
                self.bmcv.bm_image_to_tensor(output_bmimg, input_tensor)
                start_time = time.time()
                outputs = self.predict(input_tensor)
                self.inference_time += time.time() - start_time
                
        else:
            BMImageArray = eval('sail.BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(img_num):
                output_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, \
                    sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
                start_time = time.time()
                output_bmimg = self.preprocess_bmcv(bmimg_list[i], output_bmimg)
                self.preprocess_time += time.time() - start_time
                # self.preprocess_bmcv(bmimg_list[i], output_bmimg)
                bmimgs[i] = output_bmimg.data()
            input_tensor = sail.Tensor(self.handle, self.input_shape,  self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)
            start_time = time.time()
            outputs = self.predict(input_tensor)[:img_num]
            self.inference_time += time.time() - start_time

        start_time = time.time()
        res = self.postprocess(outputs)
        self.postprocess_time += time.time() - start_time

        return res

'''
队列传输bmimage + tpu 推理的压力测试，

d2d用的是gdma，，这个和tpu搬数据是串行的，
也就是会影响TPU搬数据，可以看一下满负荷的时候对推理和搬数据有多大影响
'''

    


def multidecoder(input_paths, proesss_nums, frame_count, tpu_id = 0, max_que_size = 8):
    channel_list = {}
    multiDecoder = sail.MultiDecoder(max_que_size, tpu_id)
    multiDecoder.set_local_flag(True)

    image_pipe = "/tmp/img"
    dist_pipe = "/tmp/final"
    ipc_image_pipe_list = [image_pipe+str(i) for i in range(proesss_nums)]
    ipc_dist_pipe_list = [dist_pipe+str(i) for i in range(proesss_nums)]

    if isinstance(input_paths, list):# 多路视频，str地址放在一个list
        for input_path in input_paths:
            channel_index = multiDecoder.add_channel(input_path, 1)
            print("Add Channel[{}]: {}".format(channel_index, input_path))
            channel_list[channel_index] = input_path
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

    handle = sail.Handle(tpu_id)
    bmcv = sail.Bmcv(handle)
    # image_index = 0
    time_start = time.time()
    # total_count = 0
    frame_id = 0

    
    ipc_list = [sail.IPC(True, ipc_image_pipe_list[key], ipc_dist_pipe_list[key]) for key in channel_list]
    while True:
        for key in channel_list:
            bmimg = sail.BMImage()
            ret = multiDecoder.read(int(key),bmimg,read_mode = 1) # 一次处理一个通道,read_mode阻塞的读

            # ipc = sail.IPC(True, ipc_image_pipe_list[key], ipc_dist_pipe_list[key]) # 初始化此通道的ipc
            ipc = ipc_list[key]
            if ret == 0:
                ipc.sendBMImage(bmimg, key, frame_id)
                frame_id +=1
                # total_count += 1
                # if total_count == 2000:
                #     total_time = (time.time()-time_start)*1000
                #     avg_time = total_time/total_count
                #     print("########################avg time: {:.2f}".format(avg_time))
                #     total_count = 0

            else: # 一个通道数据读完
                frame_count[int(key)] = 1
                break
        break #所有通道数据读完

'''
recive
推理,获取bmimage
'''
def preprocess_images(resnet):
    # print('---------------recive---------------------')
    handle = sail.Handle(0)
    bmcv = sail.Bmcv(handle)
    # 接收方
    ipc = sail.IPC(False, "/tmp/img", "/tmp/final",usec2c=True)

    while(True):
        # print(1)
        bmimg, channel_id, frame_id = ipc.receiveBMImage()

        output_bmimg = sail.BMImage(resnet.handle, resnet.net_h, resnet.net_w, \
        sail.Format.FORMAT_RGB_PLANAR, resnet.img_dtype)
        start_time = time.time()
        output_bmimg = resnet.preprocess_bmcv(bmimg, output_bmimg)
        resnet.preprocess_time += time.time() - start_time
        input_tensor = sail.Tensor(resnet.handle, resnet.input_shape,  resnet.input_dtype,  False, False)
        resnet.bmcv.bm_image_to_tensor(output_bmimg, input_tensor)
        start_time = time.time()
        outputs = resnet.predict(input_tensor)
        resnet.inference_time += time.time() - start_time
        res = resnet.postprocess(outputs)

        print(res)

'''
recive
一路推理,获取bmimage

'''
def multi_preprocess_images(resnet,channel_id,ipc_image_pipe_list:str,ipc_dist_pipe_list:str,frame_flag:int):
    # print('---------------recive---------------------')
    # handle = sail.Handle(0)
    # bmcv = sail.Bmcv(handle)
    ipc = sail.IPC(False, ipc_image_pipe_list, ipc_dist_pipe_list,usec2c=True) # 初始化此通道的接受ipc

    while(not frame_flag[channel_id]):
        bmimg, channel_id, frame_id = ipc.receiveBMImage()

        output_bmimg = sail.BMImage(resnet.handle, resnet.net_h, resnet.net_w, \
        sail.Format.FORMAT_RGB_PLANAR, resnet.img_dtype)
        start_time = time.time()
        output_bmimg = resnet.preprocess_bmcv(bmimg, output_bmimg)
        resnet.preprocess_time += time.time() - start_time
        input_tensor = sail.Tensor(resnet.handle, resnet.input_shape,  resnet.input_dtype,  False, False)
        resnet.bmcv.bm_image_to_tensor(output_bmimg, input_tensor)
        start_time = time.time()
        outputs = resnet.predict(input_tensor)
        resnet.inference_time += time.time() - start_time
        res = resnet.postprocess(outputs)

        print(res)
    





def main(args):

    resnet = Resnet(args)
    

    proesss_nums = 4
    input_video = [args.input for _ in range(proesss_nums)]
    frame_flag= multiprocessing.Array("i",[0 for _ in range(proesss_nums)])

    '''
    pool not work ,,need debug ...
    '''
    # # 创建解码进程池，设置进程数为4
    # with multiprocessing.Pool(processes=1) as decode_pool:
    #     # 解码进程
    #     # decode_pool.apply_async(decode_images, args=(input_video))
    #     decode_pool.map(decode_images, input_video)
    # # 创建推理进程池，设置进程数为4
    # with multiprocessing.Pool(processes=1) as preprocess_pool:
    #     # 推理进程
    #     # preprocess_pool.apply_async(preprocess_images, args=(resnet))
    #     preprocess_pool.map(preprocess_images, resnet)

    # 在外面声明？
    # ipc_send = sail.IPC(True, "/tmp/img", "/tmp/final") # 用一个dist 和 src # image pipe，final pipe
    # ipc_recive = sail.IPC(False, "/tmp/img", "/tmp/final",usec2c=True)

    '''
    not pool
    '''
    # send_process = multiprocessing.Process(target=decode_images, args=(input_video,ipc_send))
    # receive_process = multiprocessing.Process(target=preprocess_images, args=(resnet,ipc_recive))
    send_process = multiprocessing.Process(target=multidecoder, args=(input_video, proesss_nums, frame_flag, 0, 8))

    
    image_pipe = "/tmp/img"
    dist_pipe = "/tmp/final"
    ipc_image_pipe_list = [image_pipe+str(i) for i in range(proesss_nums)]
    ipc_dist_pipe_list = [dist_pipe+str(i) for i in range(proesss_nums)]

    receive_processes = [multiprocessing.Process(target=multi_preprocess_images, args=(resnet, i, ipc_image_pipe_list[i], ipc_dist_pipe_list[i], frame_flag)) for i in range(proesss_nums)]


    # 记录开始时间
    start_time = time.time()
    # 启动进程
    send_process.start()

    for receive_process in receive_processes:
        receive_process.start()

    # send_process1.start()
    # receive_process1.start()

    # send_process2.start()
    # receive_process2.start()

    # send_process3.start()
    # receive_process3.start()

    # send_process4.start()
    # receive_process4.start()
    # 等待进程结束
    send_process.join()
    for receive_process in receive_processes:
        receive_process.join()

    # send_process1.join()
    # receive_process1.join()

    # send_process2.join()
    # receive_process2.join()

    # send_process3.join()
    # receive_process3.join()
    
    # send_process4.join()
    # receive_process4.join()
    # 计算时间
    end_time = time.time()
    communication_time = end_time - start_time
    # 打印时间
    print("通信时间：", communication_time, "秒")


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    # parser.add_argument('--input', type=str, default='../../data/images/face_data_test', help='path of input, must be image directory')
    parser.add_argument('--input', type=str, default='/data/monitor.mp4', help='path of input, must be image directory')
    parser.add_argument('--bmodel', type=str, default='/workingspace/6_workspace_for_multiprocess/sample/ResNet/models/BM1684X/resnet50_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    main(args)
