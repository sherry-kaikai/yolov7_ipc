import sophon.sail as sail
import numpy as np
import threading
import time
import os
import queue
from multiprocessing import Process
import argparse
import logging
# import lprnet  
from lprnet import CHARS,CHARS_DICT

class MultiDecoderThread(object):
    def __init__(self, tpu_id, video_list, resize_type:sail.sail_resize_type, max_que_size:int, loop_count:int,process_id:int):
        self.channel_list = {}
        self.tpu_id = tpu_id
        self.break_flag = False
        self.resize_type = resize_type
        '''mytest'''
        self.resize_type_lprnet = sail.sail_resize_type.BM_RESIZE_TPU_LINEAR

        self.multiDecoder = sail.MultiDecoder(15, tpu_id)
        self.multiDecoder.set_local_flag(True)

        self.loop_count = loop_count

        self.post_que = queue.Queue(max_que_size)
        self.image_que = queue.Queue(max_que_size)

        self.exit_flag = False
        self.flag_lock = threading.Lock()
        
        self.process_id = process_id

        for video_name in video_list:
            channel_index = self.multiDecoder.add_channel(video_name,1)
            print("Process {}  Add Channel[{}]: {}".format(process_id,channel_index,video_name))
            self.channel_list[channel_index] = video_name

        self.alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)

        # lprnet 
        self.input_scale_lprnet = 1 
        # self.alpha_beta_lprnet = [
        #     x * self.input_scale_lprnet * 0.0078125 for x in [1, -127.5, 1, -127.5, 1, -127.5]
        # ]

        self.alpha_beta_lprnet = (
            tuple(x * 1 * 0.0078125 for x in [1, -127.5]),
            tuple(x * 1 * 0.0078125 for x in [1, -127.5]),
            tuple(x * 1 * 0.0078125 for x in [1, -127.5])
        )
    def get_exit_flag(self):
        self.flag_lock.acquire()
        flag_temp = self.exit_flag
        self.flag_lock.release()
        return flag_temp

    def InitProcess(self, yolo_bmodel,lprnet_bmodel,dete_threshold,nms_threshold):
        self.engine_image_pre_process = sail.EngineImagePreProcess(yolo_bmodel, self.tpu_id, 0)
        self.engine_image_pre_process.InitImagePreProcess(self.resize_type, True, 10, 10)
        self.engine_image_pre_process.SetPaddingAtrr()
        self.engine_image_pre_process.SetConvertAtrr(self.alpha_beta)
        # self.net_w = self.engine_image_pre_process.get_input_width()
        # self.net_h = self.engine_image_pre_process.get_input_height()

        output_names = self.engine_image_pre_process.get_output_names()
        self.yolo_batch_size = self.engine_image_pre_process.get_output_shape(output_names[0])[0]
        output_shapes = [self.engine_image_pre_process.get_output_shape(i) for i in output_names]
        logging.debug('Process {},YOLO RECEIVE IPC INIT DONE,bmodel output_shapes {}'.format(self.process_id,output_shapes))
        self.yolov5_post_async = sail.algo_yolov5_post_cpu_opt_async([output_shapes[0],output_shapes[1],output_shapes[2]],640,640,10)
        self.bmcv = sail.Bmcv(sail.Handle(0))

        # self.lprnet = lprnet.LPRNet(lprnet_bmodel,self.tpu_id)
        self.lprnet_engine_image_pre_process = sail.EngineImagePreProcess(lprnet_bmodel, self.tpu_id, 0)
        self.lprnet_engine_image_pre_process.InitImagePreProcess(self.resize_type_lprnet, True, 10, 10)
        self.lprnet_engine_image_pre_process.SetConvertAtrr(self.alpha_beta_lprnet)
        self.lprnet_output_names = self.lprnet_engine_image_pre_process.get_output_names()[0]

        thread_preprocess = threading.Thread(target=self.decoder_and_pushdata, args=(self.channel_list, self.multiDecoder, self.engine_image_pre_process))
        thread_inference = threading.Thread(target=self.Inferences_thread, args=(self.resize_type, self.tpu_id, self.post_que, self.image_que))
        thread_postprocess = threading.Thread(target=self.post_process, args=(self.post_que, dete_threshold, nms_threshold))
        thread_lprnet = threading.Thread(target=self.lprnet_pre_and_process,args=(self.image_que,))
        
        thread_drawresult = threading.Thread(target=self.lprnet_post_and_draw_result, args=(self.image_que,))
        
        
        thread_postprocess.start()
        thread_preprocess.start()
        thread_lprnet.start()

        thread_inference.start()
        thread_drawresult.start()
       
    
    def decoder_and_pushdata(self,channel_list, multi_decoder, PreProcessAndInference):
        image_index = 0
        time_start = time.time()
        total_count = 0
        while True:
            if self.get_exit_flag():
                    break
            for key in channel_list:
                if self.get_exit_flag():
                    break
                bmimg = sail.BMImage()
                ret = multi_decoder.read(int(key),bmimg)
                if ret == 0:
                    image_index += 1
                    PreProcessAndInference.PushImage(int(key),image_index, bmimg)
                    total_count += 1
                    if total_count == 2000:
                        total_time = (time.time()-time_start)*1000
                        avg_time = total_time/total_count
                        print("########################avg time: {:.2f}".format(avg_time))
                        total_count = 0

                else:
                    time.sleep(0.01)

        print("decoder_and_pushdata thread exit!")

    def Inferences_thread(self, resize_type:sail.sail_resize_type, device_id:int, post_queue:queue.Queue, img_queue:queue.Queue):
        while True:
            if self.get_exit_flag():
                break
            start_time = time.time()
            output_tensor_map, ost_images, channel_list ,imageidx_list, padding_atrr = self.engine_image_pre_process.GetBatchData(True)

            # print('111111111111111111111111111111111')
            # print('pre get channel_list ,imageidx_list,',channel_list ,imageidx_list)
            # print('111111111111111111111111111111111')

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
            post_queue.put([output_tensor_map,
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
                # print('00000000000000000000000000000000000 img queue')
                # print('put to img queue',(channel,imageidx_list[index]))
                # print('00000000000000000000000000000000000')
                img_queue.put({(channel,imageidx_list[index]):ost_images[index]}) 
                
                logging.debug("put ost img to queue,  cid is {},frameid is{}".format(channel,imageidx_list[index]))

            # elements = list(img_queue.queue)
            # print("ost put ~~~~~~~~~!!!!!!!!!!!!!!!!!!!!!")
            # print(elements)
            # print("ost put ~~~~~~~~~!!!!!!!!!!!!!!!!!!!!!")
            end_time = time.time()
            logging.info("Engine_image_pre_process GetBatchData time use: {:.2f} ms".format((end_time-start_time)*1000))
        
        print("Inferences_thread thread exit!")


    def post_process(self, post_quque:queue.Queue, dete_threshold:float, nms_threshold:float):
        while (True):
            if self.get_exit_flag():
                break
            if post_quque.empty():
                time.sleep(0.01)
                continue
            output_tensor_map, channels ,imageidxs, ost_ws, ost_hs, padding_atrrs = post_quque.get(True)

            dete_thresholds = np.ones(len(channels),dtype=np.float32)
            nms_thresholds = np.ones(len(channels),dtype=np.float32)
            dete_thresholds = dete_threshold*dete_thresholds
            nms_thresholds = nms_threshold*nms_thresholds
            while True:
                if self.get_exit_flag():
                    break
                ret = self.yolov5_post_async.push_data(channels, imageidxs, output_tensor_map, dete_thresholds, nms_thresholds, ost_ws, ost_hs, padding_atrrs)
                # print('33333333333333333333333333333333')
                # print("yolo post push",channels, imageidxs) # 判断，看上去不是一一对应？ 需要用queue吗
                # print('3333333333333333333333333333333333')
                if ret == 0:
                    break
                else:
                    print("push_data failed, ret: {}".format(ret))
                    time.sleep(0.01)
                break
        print("post_process thread exit!")
    
    def lprnet_pre_and_process(self, img_queue:queue.Queue):
        while (True):
            if self.get_exit_flag():
                break
            if img_queue.empty():
                time.sleep(0.01)
                continue

            # elements = list(img_queue.queue)
            # print(elements)
            ocv_image = img_queue.get(True) 

            objs, channel, image_idx = self.yolov5_post_async.get_result_npy() #yolov5推理的单输出结果。和ocv_image 如何一一对应？queue中维护一个dict？

            print("lprnet_pre_and_process: yolo post id and ocv id is ",(channel,image_idx),ocv_image.keys()) # 判断，看上去不是一一对应？ 需要用queue吗

            if (channel,image_idx) == list(ocv_image.keys())[0]:
                for obj in objs: # 一张图上多个结果
                    x1, y1, x2, y2, category_id, score = obj

                    logging.debug("Lprnet_pre_and_process:Process {},channel_idx is {} image_idx is {},len(objs) is{}".format(self.process_id,channel, image_idx, len(objs)))
                    logging.info("Lprnet_pre_and_process:Process %d,YOLO postprocess DONE! objs:tuple[left, top, right, bottom, class_id, score] :%s",self.process_id,obj)

                    croped = self.bmcv.crop(list(ocv_image.values())[0],int(x1),int(y1),int(x2-x1),int(y2-y1))
                    self.lprnet_engine_image_pre_process.PushImage(channel, image_idx, croped)
            else:

                logging.error("lprnet_pre_and_process:  yolo post result idx{}is not equal to origin images idx{}")
                logging.error((channel,image_idx) ,list(ocv_image.keys())[0])

        print("Lprnet_pre_and_process thread exit!")

    def lprnet_post_and_draw_result(self, img_queue:queue.Queue):

        total_count = 0
        start_time = time.time()
        while (True):
            if img_queue.empty():
                time.sleep(0.01)
                continue
            # ocv_image = img_queue.get(True) 
            # 1 get lprnet process res
            
            output, _, channel_list, image_idx_list,_ = self.lprnet_engine_image_pre_process.GetBatchData_Npy() 

            print(type(output),output.keys(),len(output),self.lprnet_output_names,channel_list, image_idx_list)


            # output_array = output[self.lprnet_output_names]
            output_array = output[self.lprnet_output_names][:4]
            print(len(output_array))

            res = list()
            for temp in np.argmax(output_array, axis=1):
                no_repeat_blank_label = list()
                pre_c = temp[0]
                if pre_c != len(CHARS) - 1:
                    no_repeat_blank_label.append(CHARS_DICT[pre_c])
                for c in temp:
                    if (pre_c == c) or (c == len(CHARS) - 1):
                        if c == len(CHARS) - 1:
                            pre_c = c
                        continue
                    no_repeat_blank_label.append(CHARS_DICT[c])
                    pre_c = c
                res.append("".join(no_repeat_blank_label))
            print(res)
            logging.info('LPRNET POSTPROCESS DONE,res{}'.format(res))

            '''draw'''
                # bmcv.rectangle(ocv_image, obj[0], obj[1], obj[2]-obj[0], obj[3]-obj[1],(0,0,255),2)
            # image = sail.BMImage(handle,ocv_image.height(),ocv_image.width(),sail.Format.FORMAT_YUV420P,sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
            # bmcv.convert_format(ocv_image,image)
            # for obj in objs:
            #     txt_d = "{}".format(obj[4])
            #     bmcv.putText(image, txt_d , obj[0], obj[1], [0,0,255], 1.4, 2)
            # bmcv.imwrite("{}_{}.jpg".format(channel,image_idx),image)

            
            if self.loop_count <=  image_idx_list[-1]:
                logging.debug("LOOPS DONE")
                break
        end_time = time.time()
        time_use = (end_time-start_time)*1000
        avg_time = time_use/image_idx_list[-1]

        print("Process {}:Total images: {} ms".format(self.process_id, self.loop_count))
        print("Total time use: {} ms".format(time_use))
        print("Avg time use: {} ms".format(avg_time))
        print("Process {}: {} FPS".format(self.process_id, 1000/avg_time))
        print("Result thread exit!")

        logging.info("Process {}:Loops{},Total time use: {} ms, avg_time{}, {} FPS".format(self.process_id, self.loop_count,time_use,avg_time,1000/avg_time))

        
        self.flag_lock.acquire()
        self.exit_flag = True
        self.flag_lock.release()

def process_demo(tpu_id, max_que_size, video_name_list, yolo_bmodel,lprnet_bmodel, loop_count, process_id,dete_threshold,nms_threshold):
    process =  MultiDecoderThread(tpu_id, video_name_list, sail.sail_resize_type.BM_PADDING_TPU_LINEAR, max_que_size, loop_count,process_id)
    process.InitProcess(yolo_bmodel,lprnet_bmodel,dete_threshold,nms_threshold)


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--multidecode_max_que_size', type=int, default=16, help='multidecode queue')
    parser.add_argument('--ipc_recive_queue_len', type=int, default=16, help='ipc recive queue')
    parser.add_argument('--video_nums', type=int, default=16, help='procress nums of input')
    parser.add_argument('--batch_size', type=int, default=4, help='video_nums/batch_size is procress nums of process and postprocess')
    parser.add_argument('--loops', type=int, default=100, help='process loops for one video')
    parser.add_argument('--input', type=str, default='/data/licenseplate_640516-h264.mp4', help='path of input, must be video path') 
    parser.add_argument('--yolo_bmodel', type=str, default='../models/yolov5s-licensePLate/BM1684/yolov5s_v6.1_license_3output_int8_4b.bmodel', help='path of bmodel')
    parser.add_argument('--lprnet_bmodel', type=str, default='../models/lprnet/BM1684/lprnet_int8_4b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = argsparser()

    logging.basicConfig(filename= f'1684_yolo_process_and_video_thread_is_{args.video_nums}.log',filemode='w',level=logging.DEBUG)

    # decoder_count = 4           #每个进程解码的路数
    max_que_size = args.multidecode_max_que_size           #缓存的大小
    loop_count = args.loops           #每个进程处理图片的数量，处理完毕之后会退出。

    
    '''
    多路视频流
    '''
    process_nums = int(args.video_nums/args.batch_size)
    input_videos = [args.input for _ in range(int(args.video_nums/process_nums))]

    dete_threshold,nms_threshold = 0.65,0.65

    decode_yolo_processes = [Process(target=process_demo,args=(args.dev_id, max_que_size, input_videos, args.yolo_bmodel,args.lprnet_bmodel, loop_count, i,dete_threshold,nms_threshold)) for i in range(process_nums) ]
    for i in decode_yolo_processes:
        i.start()
        logging.debug('start decode and yolo process')
    start_time = time.time()

    logging.info(start_time)
    for i in decode_yolo_processes:
        i.join()
        logging.debug('DONE decode and yolo process')

    total_time = time.time() - start_time
    logging.info('video nums{}, process is {},total time is {},loops for one process is {},total fps is {}'.format(args.video_nums,process_nums,total_time,loop_count,(loop_count*process_nums)/total_time))

