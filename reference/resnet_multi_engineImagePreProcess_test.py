# 本例程适用于单芯片设备实现多模型的推理

import argparse
import sophon.sail as sail
import numpy as np
# import threading
import time
import os
import cv2
import queue
from multiprocessing import Process
import multiprocessing
import logging
logging.basicConfig(level=logging.INFO)



class MultiDecoderThread(object):
    def __init__(self, tpu_id, bmodel_name,input_paths, resize_type:sail.sail_resize_type, max_que_size:int, loop_count:int, process_threads:int):
        self.channel_list = {}
        self.tpu_id = tpu_id
        self.break_flag = False
        self.resize_type = resize_type
        self.bmodel_name = bmodel_name
        self.multiDecoder = sail.MultiDecoder(max_que_size, tpu_id)
        self.multiDecoder.set_local_flag(True)
        self.loop_count = loop_count
        self.process_threads = process_threads

        self.flag = [True for _ in range(self.process_threads)]
        # 原本这里初始化了多个不同的queue队列，作为不同路推理的数据传输
        # self.post_que_list = [queue.Queue(max_que_size) for i in range(self.bmodel_num)] 
        # self.image_que_list = [queue.Queue(max_que_size) for i in range(self.bmodel_num)]

        '''
        初始化ipc队列，传输bmimage
        '''
        # ERROR ， sail.ipc不能作为传入多进程的函数
        # self.ipc_list = [sail.IPC(True, image_pipe+str(i), dist_pipe+str(i)) for i in range(self.process_threads)]
        
        # may work
        # 只初始化一个ipc_list队列，里面有多个sail.ipc的参数名字
        image_pipe = "/tmp/img"
        dist_pipe = "/tmp/final"
        self.ipc_image_pipe_list = [image_pipe+str(i) for i in range(self.process_threads)]
        self.ipc_dist_pipe_list = [dist_pipe+str(i) for i in range(self.process_threads)]

        self.exit_flag = False
    
        if os.path.isfile(input_paths): # 单视频
            # print('************************is singel video************************')
            # multidecoder
            channel_index = self.multiDecoder.add_channel(input_paths,1)
            print("Add Channel[{}]: {}".format(channel_index,input_paths))
            self.channel_list[channel_index] = input_paths
        elif isinstance(input_paths, list):# 多路视频，str地址放在一个list
            for input_path in input_paths:
                channel_index = self.multiDecoder.add_channel(input_path, 1)
                print("Add Channel[{}]: {}".format(channel_index, input_path))
                self.channel_list[channel_index] = input_path
        elif os.path.isdir(input_paths): # 文件夹
            for root, dirs, filenames in os.walk(input_paths):
                for filename in filenames:
                    print('************************',filename,'************************')
                    img_file = os.path.join(root, filename)
                    # TODO

                    
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.a = [1/(255.*x) for x in self.std]
        self.b = [-x/y for x,y in zip(self.mean, self.std)]
        self.alpha_beta = []
        for i in range(3):
            self.alpha_beta.append(self.a[i]*self.input_scale)
            self.alpha_beta.append(self.b[i]*self.input_scale)
    
    # 这里是对多线程
    # def get_exit_flag(self):
    #     self.flag_lock.acquire()
    #     flag_temp = self.exit_flag
    #     self.flag_lock.release()
    #     return flag_temp



    # 初始化并开启进程池
    def InitProcess(self,  process_id):
        self.process_id = process_id
        # self.engine_image_pre_process_list = []
        # self.EngineImagePreProcess(self.bmodel_name)

        # 带有预处理功能的图像推理接口，内部使用线程池的方式，Python下面有更高的效率
        self.engine_image_pre_infer_process = sail.EngineImagePreProcess(self.bmodel_name, self.tpu_id, 0) 
        self.engine_image_pre_infer_process.InitImagePreProcess(self.resize_type, True, 10, 10)
        self.engine_image_pre_infer_process.SetPaddingAtrr()
        self.engine_image_pre_infer_process.SetConvertAtrr(self.alpha_beta)
        self.net_w = self.engine_image_pre_infer_process.get_input_width()
        self.net_h = self.engine_image_pre_infer_process.get_input_height()
        output_name = self.engine_image_pre_infer_process.get_output_names()[0]
        self.batch_size = self.engine_image_pre_infer_process.get_output_shape(output_name)[0]
        '''
        threading.Thread
        '''
        # thread_preprocess = threading.Thread(target=self.decoder_and_pushdata, args=(self.channel_list, self.multiDecoder, self.engine_image_pre_process_list)) # 调用sail接口，一步进行多线程解码和预处理
        # thread_inference_list = [threading.Thread(target=self.Inferences_thread, args=(idx, self.resize_type, self.tpu_id, self.post_que_list[idx], self.image_que_list[idx])) for idx in range(self.process_threads)]
        # thread_postprocess_list = [threading.Thread(target=self.post_process, args=(idx, self.post_que_list[idx], 0.2, 0.5)) for idx in range(self.process_threads)]
        # # thread_drawresult_list = [threading.Thread(target=self.draw_result_thread, args=(idx, self.image_que_list[idx],)) for idx in range(self.bmodel_num)]
        # for i in range(self.process_threads):
        #     # thread_drawresult_list[i].start()
        #     thread_postprocess_list[i].start()
        #     thread_inference_list[i].start()
        # thread_preprocess.start()

        '''
        multiprocessing.Process, in list 
        Pool?
        问题是，ipc初始化后不能作为参数传入multiprocessing.Process中,需要在每个进程中初始化
        '''
        process_preprocess = Process(target=self.decoder_and_pushdata, args=(self.channel_list, self.multiDecoder, self.engine_image_pre_infer_process)) # 调用sail接口，一步进行多线程解码和  预处理推理初始化
        process_inference_list = [Process(target=self.Inferences_process, args=(idx, self.resize_type, self.tpu_id, self.ipc_image_pipe_list[idx], self.ipc_dist_pipe_list[idx])) for idx in range(self.process_threads)]

        for i in range(self.process_threads):
            process_inference_list[i].start()
        for i in range(self.process_threads):
            process_inference_list[i].join()

        process_preprocess.start() # 为什么放在处理的开启后面？
        process_preprocess.join()
        
    # 1.如何告诉队列读取结束？ class加一个flag参数？
    # 2.捋一下解码-预处理推理的逻辑
    def decoder_and_pushdata(self, channel_list, multiDecoder, engine_image_pre_infer_process):
        handle = sail.Handle(self.tpu_id)
        bmcv = sail.Bmcv(handle)
        image_index = 0
        time_start = time.time()
        total_count = 0
        while True:
            for key in channel_list:
                bmimg = sail.BMImage()
                ret = multiDecoder.read(int(key),bmimg,read_mode=1) # 一次处理一个通道
                if ret == 0:
                    # 在此处进行数据的分发,多次预处理和推理 
                    for i in range(self.process_threads):
                        if i >=1 :
                            bmimg_temp = sail.BMImage(handle,bmimg.height(),bmimg.width(),bmimg.format(),bmimg.dtype())
                            bmcv.image_copy_to(bmimg,bmimg_temp,0,0)
                            image_index += 1
                            engine_image_pre_infer_process[i].PushImage(int(key),image_index, bmimg_temp)
                    
                    image_index += 1
                    engine_image_pre_infer_process[0].PushImage(int(key),image_index, bmimg)
                    
                    total_count += 1
                    if total_count == 2000:
                        total_time = (time.time()-time_start)*1000
                        avg_time = total_time/total_count
                        print("########################avg time: {:.2f}".format(avg_time))
                        total_count = 0

                else: # 一个通道数据读完
                    self.flag[int(key)] = False
                    break
            break #所有通道数据读完

        print("decoder_and_pushdata thread exit!")

    def Inferences_process(self, process_idx, resize_type:sail.sail_resize_type, device_id:int, post_queue:queue.Queue, img_queue:queue.Queue):
        while True:
            if self.get_exit_flag():
                break
            start_time = time.time()
            output_tensor_map, ost_images, channel_list ,imageidx_list, padding_atrr = self.engine_image_pre_infer_process[process_idx].GetBatchData(True)
            tensor_with_name = output_tensor_map[0]
            width_list = []
            height_list= []
            for index, channel in enumerate(channel_list):
                width_list.append(ost_images[index].width())
                height_list.append(ost_images[index].height())

            while post_queue.full():
                time.sleep(0.01)
                if self.get_exit_flag():
                    break
                continue
            post_queue.put([tensor_with_name,
                            channel_list,
                            imageidx_list,
                            width_list, 
                            height_list, 
                            padding_atrr],False)
            
            for index, channel in enumerate(channel_list):
                if self.get_exit_flag():
                    break
                while img_queue.full():
                    time.sleep(0.01)
                    if self.get_exit_flag():
                        break
                    continue 
                img_queue.put(ost_images[index])

            end_time = time.time()
            print("GetBatchData time use: {:.2f} ms".format((end_time-start_time)*1000))
        
        print("Inferences_thread thread exit!")


    def post_process(self, thread_idx, post_quque:queue.Queue, dete_threshold:float, nms_threshold:float):
        while (True):
            if self.get_exit_flag():
                break
            if post_quque.empty():
                time.sleep(0.01)
                continue
            tensor_with_name, channels ,imageidxs, ost_ws, ost_hs, padding_atrrs = post_quque.get(True)
            dete_thresholds = np.ones(len(channels),dtype=np.float32)
            nms_thresholds = np.ones(len(channels),dtype=np.float32)
            dete_thresholds = dete_threshold*dete_thresholds
            nms_thresholds = nms_threshold*nms_thresholds
            while True:
                if self.get_exit_flag():
                    break
                ret = self.yolov5_post_list[thread_idx].push_data(channels, imageidxs, tensor_with_name, dete_thresholds, nms_thresholds, ost_ws, ost_hs, padding_atrrs)
                if ret == 0:
                    break
                else:
                    print("push_data failed, ret: {}".format(ret))
                    time.sleep(0.01)
                break
        print("post_process thread exit!")
    
    def draw_result_thread(self, thread_idx, img_queue:queue.Queue):
        handle = sail.Handle(self.tpu_id)
        bmcv = sail.Bmcv(handle)
        total_count = 0
        color_list = [
            [0,0,255],
            [0,255,0],
            [255,0,0],
        ]
        start_time = time.time()
        while (True):
            if img_queue.empty():
                time.sleep(0.01)
                continue
            ocv_image = img_queue.get(True)
            objs, channel, image_idx = self.yolov5_post_list[thread_idx].get_result_npy()
            for obj in objs:
                bmcv.rectangle(ocv_image, obj[0], obj[1], obj[2]-obj[0], obj[3]-obj[1],color_list[thread_idx%len(color_list)],2)
            image = sail.BMImage(handle,ocv_image.height(),ocv_image.width(),sail.Format.FORMAT_YUV420P,sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
            bmcv.convert_format(ocv_image,image)
            for obj in objs:
                txt_d = "{}".format(obj[4])
                bmcv.putText(image, txt_d , obj[0], obj[1], color_list[thread_idx%len(color_list)], 1.4, 2)
            bmcv.imwrite("./images/thread{}_channel{}_idx{}.jpg".format(thread_idx,channel,image_idx),image)

            total_count += 1
            if self.loop_count <=  total_count:
                break
        end_time = time.time()
        time_use = (end_time-start_time)*1000
        avg_time = time_use/self.loop_count

        print("Total images: {}".format(self.loop_count))
        print("Total time use: {}".format(time_use))
        print("Avg time use: {}".format(avg_time))
        print("{}: {} PFS".format(self.process_id, 1000/avg_time))
        print("Result thread exit!")
        
        self.flag_lock.acquire()
        self.exit_flag = True
        self.flag_lock.release()

def process_demo(tpu_id, bmodel_name, max_que_size, video_name_list, loop_count, process_id):
    process = MultiDecoderThread(tpu_id, bmodel_name, video_name_list, sail.sail_resize_type.BM_RESIZE_TPU_NEAREST, max_que_size, loop_count)
    process.InitProcess(process_id)

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    # parser.add_argument('--input', type=str, default='../../data/images/face_data_test', help='path of input, must be image directory')
    parser.add_argument('--input', type=str, default='/data/monitor.mp4', help='path of input, must be image directory')
    parser.add_argument('--bmodel', type=str, default='/workingspace/sophon-demo/sample/ResNet/models/BM1684X/resnet50_int8_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args



def main(args):
    tpu_id = 0
    decoder_count = 4
    max_que_size = 10
    loop_count = 1000 # 跑1000张图 也可以等视频结束就结束测试？

    # video_path = "./video"
    # video_name_list = [ "001.mp4",
    #                     "002.mp4"]

    input_videos = args.input

    bmodel_name = args.bmodel
    process_demo(tpu_id,bmodel_name, max_que_size, input_videos, loop_count, 1001)
    
    # p0 = Process(target=process_demo, args=(tpu_id, max_que_size, input_videos, loop_count, 1001)) # 1001是进程号
    # p1 = Process(target=process_demo, args=(tpu_id, max_que_size, video_list, bmodel_name, loop_count, 1002))
    # p2 = Process(target=process_demo, args=(tpu_id, max_que_size, video_list, bmodel_name, loop_count, 1003))
    # p0.start()
    # p1.start()
    # p2.start()


if __name__ == '__main__':
    args = argsparser() 
    main(args)


