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

class multidecoder_Yolov5(object):
    def __init__(self, tpu_id, process_id, video_paths, model_path, loops:int = 100, max_que_size:int = 4, dete_threshold = 0.65, nms_threshold = 0.65):
        self.process_id = process_id
        self.frame_id = 0   
        self.loops = loops*len(video_paths)
        self.channel_list = {}
        self.multiDecoder = sail.MultiDecoder(max_que_size, tpu_id)
        self.multiDecoder.set_local_flag(True)
        self.multiDecoder.set_read_timeout(1) # 设置超时时间1s
        logging.info('multidecoder INIT DONE')


        if isinstance(video_paths, list):
            for input_path in video_paths:
                channel_index = self.multiDecoder.add_channel(input_path) # 不丢帧
                logging.info("Process {} ,Add Channel[{}]: {}".format(self.process_id,channel_index, input_path))
                self.channel_list[channel_index] = input_path

        self.stop_signal = False

        resize_type = sail.sail_resize_type.BM_PADDING_TPU_LINEAR # 需要选择正确的resizetype，否则检测结果错误
        alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)

        self.engine_image_pre_process = sail.EngineImagePreProcess(model_path, tpu_id, use_mat_output=0) # use_mat_output 是否使用OpenCV的Mat作为图片的输出，默认为False，不使用。
        self.engine_image_pre_process.InitImagePreProcess(resize_type, True, 10, 10)
        self.engine_image_pre_process.SetPaddingAtrr()
        self.engine_image_pre_process.SetConvertAtrr(alpha_beta)
        net_w = self.engine_image_pre_process.get_input_width()
        net_h = self.engine_image_pre_process.get_input_height()


        output_names = self.engine_image_pre_process.get_output_names()
        self.batch_size = self.engine_image_pre_process.get_output_shape(output_names[0])[0]
        output_shapes = [self.engine_image_pre_process.get_output_shape(i) for i in output_names]
        logging.debug('Process {},YOLO RECEIVE IPC INIT DONE,bmodel output_shapes {}'.format(self.process_id,output_shapes))
        
        self.dete_thresholds = dete_threshold*np.ones(self.batch_size,dtype=np.float32)
        self.nms_thresholds = nms_threshold*np.ones(self.batch_size,dtype=np.float32)

        self.yolov5_post_async = sail.algo_yolov5_post_cpu_opt_async([output_shapes[0],output_shapes[1],output_shapes[2]],640,640,10)
        self.bmcv = sail.Bmcv(sail.Handle(0))

    def multidecoder_pushdata(self):
        '''
        功能：
            multi解码多路视频流, 将解码后的数据推入sail.EngineImagePreProcess（预处理和推理）接口。
        '''
        for key in self.channel_list:
            bmimg = sail.BMImage()
            ret = self.multiDecoder.read(int(key), bmimg, read_mode = 1)   
            if ret == 0:
                self.frame_id +=1
                ret = self.engine_image_pre_process.PushImage(key, self.frame_id, bmimg)
                if ret == 0:
                    logging.info("Process %d,decode channel and sent to engine_image_pre_process done, channle id is %d, frameid is %d",self.process_id,key,self.frame_id)
                else:
                    logging.error("Process %d,sent to engine_image_pre_process ERROR, channle id is %d, frameid is %d",self.process_id,key,self.frame_id)
                    time.sleep(0.001)
            else: 
                time.sleep(0.001)

    def inference_and_postprocess(self):
        '''
        功能：
            从sail.EngineImagePreProcess中得到推理完成的一个batch的数据，随后进行后处理，最后crop。
            
        返回值：
            [{(channel_idx, image_idx):img},{(channel_idx, image_idx):img}]
        '''
        get_batch_data_time = time.time()

        # 0 获取推理后的batch数据
        output_tensor_map, ost_images, channel_list, imageidx_list, padding_atrr = self.engine_image_pre_process.GetBatchData(True) # d2s
        logging.info("Process {},YOLO pre and process done,with out d2s, get one batch data time use: {:.4f}s".format(self.process_id,time.time()-get_batch_data_time))

        # 处理数据
        width_list = []
        height_list= []
        image_dict = {}
        for i in range(len(ost_images)):
            width_list.append(ost_images[i].width())
            height_list.append(ost_images[i].height())
            image_dict[(channel_list[i],imageidx_list[i])] = ost_images[i]

        
        # 1 后处理
        start_time = time.time()
        ret = self.yolov5_post_async.push_data(channel_list, imageidx_list, output_tensor_map, self.dete_thresholds, self.nms_thresholds, width_list, height_list, padding_atrr)
        if ret == 0:
            logging.debug("Process {},push data to YOLO postprocess SUCCESS, ret: {}".format(self.process_id,ret))
        else:
            logging.error("Process {},push data to YOLO postprocess FAIL, ret: {}".format(self.process_id,ret))
            time.sleep(0.001)
        logging.info("Process {},YOLO postprocess push data done, time use: {:.2f}s".format(self.process_id,time.time() - start_time))

        # 2 得到后处理的一张图的结果，并做crop
        crop_time = time.time()
        res = []
        for _ in range(self.batch_size):
            objs, channel_idx, image_idx = self.yolov5_post_async.get_result_npy() # objs:tuple[left, top, right, bottom, class_id, score] 一张图上的多个检测框

            boxes = []
            for idx in range(len(objs)):
                x1, y1, x2, y2, category_id, score = objs[idx]
                # bbox_dict = [align(x1,2),align(y1,2),align((x2-x1),2),align((y2-y1),2)]
                bbox_dict = [int(x1),int(y1),int(x2-x1),int(y2-y1)]
                boxes.append(bbox_dict)
                logging.debug("Process {},channel_idx is {} image_idx is {},len(objs) is{}".format(self.process_id,channel_idx, image_idx, len(objs)))
                logging.debug(bbox_dict)
                logging.info("Process %d,YOLO postprocess DONE! objs:tuple[left, top, right, bottom, class_id, score] :%s",self.process_id,objs[idx])

            # crop
            croped_list = []
            for box in boxes:
                croped_list.append(self.bmcv.crop(image_dict[(channel_idx,image_idx)],box[0],box[1],box[2],box[3]))

            logging.info("Process %d,image {} CROP DONE! ".format(image_idx))
            for img in croped_list:
                res.append({(channel_idx, image_idx):img})
            
        logging.info("Process {},YOLO crop all done, time use: {:.2f}s".format(self.process_id,time.time() - crop_time))
        return res 

    def process(self):

        while not self.stop_signal:
            self.multidecoder_pushdata()
            croped_list = self.inference_and_postprocess()
            logging.info("Process %d,YOLO process done, %s,",self.process_id,croped_list)
            logging.info('Process {},!!!!!!!!!process {} loops done,total loops is{}'.format(self.process_id,self.frame_id,self.loops))

            if self.frame_id == self.loops:
                self.stop_signal = True
                logging.info('Process {},!!!!!!!!!process {} loops done,total loops is{}'.format(self.process_id,self.frame_id,self.loops))
                break
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



def start(tpuid,process_id,input_videos,yolo_bmodel,loops,multidecode_max_que_size,dete_threshold,nms_threshold):
    decode_yolo = multidecoder_Yolov5(tpuid,process_id,input_videos,yolo_bmodel,loops,multidecode_max_que_size,dete_threshold,nms_threshold)
    decode_yolo.process()

def main(args):
    try:

        '''
        多路视频流
        '''
        process_nums = int(args.video_nums/args.batch_size)
        input_videos = [args.input for _ in range(process_nums)]

        
        

        '''
        test1
        '''
        # start(args.dev_id,0,input_videos, args.yolo_bmodel,args.loops,args.multidecode_max_que_size,0.65,0.65)

        '''
        test2 多进程
        '''
        # decode_yolo = multidecoder_Yolov5(0,input_videos,args.yolo_bmodel,args.loops,args.multidecode_max_que_size,0.65,0.65)
        # decode_yolo_processes = [multiprocessing.Process(target=decode_yolo.process) for i in range(process_nums) ]
        
        '''
        test3 class 放在外面好像不行？
        '''
        decode_yolo_processes = [multiprocessing.Process(target=start,args=(args.dev_id,i,input_videos, args.yolo_bmodel,args.loops,args.multidecode_max_que_size,0.65,0.65)) for i in range(process_nums) ]
        for i in decode_yolo_processes:
            i.start()
            logging.debug('start decode and yolo process')
        start_time = time.time()
        logging.info(start_time)
        for i in decode_yolo_processes:
            i.join()
            logging.debug('DONE decode and yolo process')
        
        

        total_time = time.time() - start_time
        logging.info('total time {}'.format(total_time))
        logging.info('loop is %d ,total frame is %d, fps is %.4f'% (args.loops,args.loops*args.video_nums,(args.loops*args.video_nums)/total_time))
        print('loop is %d ,total frame is %d, fps is %.4f'% (args.loops,args.loops*args.video_nums,(args.loops*args.video_nums)/total_time))
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
    parser.add_argument('--multidecode_max_que_size', type=int, default=16, help='multidecode queue')
    parser.add_argument('--ipc_recive_queue_len', type=int, default=16, help='ipc recive queue')
    parser.add_argument('--chip_mode', type=str, default='1684x', help='1684x or 1684')
    parser.add_argument('--video_nums', type=int, default=16, help='procress nums of input')
    parser.add_argument('--batch_size', type=int, default=4, help='video_nums/batch_size is procress nums of process and postprocess')
    parser.add_argument('--loops', type=int, default=100, help='process loops for one video')
    parser.add_argument('--input', type=str, default='/data/licenseplate_640516-h264.mp4', help='path of input, must be video path') 
    parser.add_argument('--yolo_bmodel', type=str, default='../models/yolov5s-licensePLate/BM1684X/yolov5s_v6.1_license_3output_int8_4b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()

    if args.chip_mode == '1684':
        args.yolo_bmodel = '../models/yolov5s-licensePLate/BM1684/yolov5s_v6.1_license_3output_int8_4b.bmodel'
    logging.basicConfig(filename= f'{args.chip_mode}_without_ipc_process_is_{args.video_nums}.log',filemode='w',level=logging.DEBUG)
    try:
        main(args)
    except Exception as e:
        print("An exception occurred:", e)
        # 打印异常堆栈信息
        traceback.print_exc()
        # 可以选择退出程序
        sys.exit(1)

