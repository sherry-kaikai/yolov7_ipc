from sophon import sail
import time
import logging
import numpy as np
logging.getLogger()   
import sys
import traceback



class multidecoder_Yolov5(object):
    def __init__(self, tpu_id, process_id, video_paths, model_path, ipc_image_pipe_name:str, ipc_dist_pipe_name:str, loops:int = 100, 
                 max_que_size:int = 16, ipc_recive_queue_len:int = 16,post_queue_len:int=10,dete_threshold = 0.65, nms_threshold = 0.65,yolov5_network_w:int=640,yolov5_network_h:int=640):
        
        self.receive_ipc = sail.IPC(False, ipc_image_pipe_name, ipc_dist_pipe_name, True, ipc_recive_queue_len) # ,10
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

        self.yolov5_post_async = sail.algo_yolov5_post_cpu_opt_async([output_shapes[0],output_shapes[1],output_shapes[2]],yolov5_network_w,yolov5_network_h,post_queue_len)
        self.bmcv = sail.Bmcv(sail.Handle(0))

    
    def align(num, align_to):
        align_to = align_to - 1
        return  (num + align_to) & (~align_to)

    def yolov5_pre_infer(self, channel_id_list:list, frame_id_list:list, bmi_list:list):
        '''
        功能：
            通过sail.Engine_image_pre_process对传入的传入长度为4的bmimage_list进行预处理和推理处理。
        
        参数：
            engine_image_pre_process: This is a sail method that has been initialized
            channel_id_list:传入长度为4的bmimage_list的通道号
            frame_id_list:传入长度为4的bmimage_list的图像号
            bmi_list:传入长度为4的bmimage
        '''
        time_start = time.time()
        
        for i in range(len(bmi_list)):
            bmi = bmi_list[i]
            logging.debug("pushdata start, bmimg w is {}, h is {}, len(bm_list) is {}".format(bmi.width(),bmi.height(),len(bmi_list)) )
            ret = self.engine_image_pre_process.PushImage(channel_id_list[i], frame_id_list[i], bmi)
            while(ret != 0):
                logging.debug("Porcess[{}]:[{}]PreProcessAndInference Thread Full, sleep: 10ms!".format(channel_id_list[i],frame_id_list[i]))
                time.sleep(0.01)

        logging.info("pushdata exit, time use: {:.4f}s".format(time.time()-time_start))

        get_batch_data_time = time.time()
        res = self.engine_image_pre_process.GetBatchData(True)   # d2s

        logging.info("YOLO pre and process done,with out d2s, get one batch data time use: {:.4f}s".format(time.time()-get_batch_data_time))
        logging.info("YOLO pre and process done,with out d2s, total time use: {:.4f}s".format(time.time()-time_start))

        return  res # (output_tensor_map, ost_images, channel_list, imageidx_list, padding_atrr)



    def yolov5_post(self, output_tensor_map:list, ost_images:list, channel_list:list, imageidx_list:list, padding_atrr:list, batch_size:int):
        '''
        功能：
            调用sail.algo_yolov5_post_cpu_opt_async 进行批后处理，一次处理一个batch的图片，并将检测出的小图从原图中crop出来

        输入：
            yolo_output_shape: bmodel的输出shape，用于后处理
            output_tensor_map: engine_image_pre_process.GetBatchData(True) 输出的第一个数据，即为4batch的推理结果 
                (Pdb) p output_tensor_map[0].get_name()
                '364_Transpose_f32'
                (Pdb) p output_tensor_map[0].get_data()
                <sophon.sail.Tensor object at 0x7f4625f3f5f0>
                (Pdb) p output_tensor_map[1].get_name()
                '381_Transpose_f32'
                (Pdb) p output_tensor_map[2].get_name()
                'output0_Transpose_f32'

            ost_images:list[BMImage]: 原始图片序列 
            channel_list:list[int]: 结果对应的原始图片的通道序列。
            imageidx_list:list[int]:结果对应的原始图片的编号序列。
            padding_atrr:list[list[int]]:填充图像的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度
            batch_size:int: 输入batch，确保输入的output_tensor_map的长度是bmodel的一个batch
            dete_threshold = 0.01:
            nms_threshold = 0.6:
            tpu_id = 0:

        返回值：
            裁剪后的小图 格式为list[dict{(cid,frame_id):croped_images}]    (？这里要返回原始大图吗？)
            
        注意：
            需要cid、imageid 来一一对应做crop的原图
        '''

        start_time = time.time()

        # 0 处理数据
        width_list = []
        height_list= []
        image_dict = {}
        for i in range(len(ost_images)):
            width_list.append(ost_images[i].width())
            height_list.append(ost_images[i].height())
            image_dict[(channel_list[i],imageidx_list[i])] = ost_images[i]


        logging.info("YOLO postprocess init done, time use: {:.2f}s".format(time.time() - start_time))


        # 1 后处理
        ret = self.yolov5_post_async.push_data(channel_list, imageidx_list, output_tensor_map, self.dete_thresholds, self.nms_thresholds, width_list, height_list, padding_atrr)
        if ret == 0:
            logging.debug("push data to YOLO postprocess SUCCESS, ret: {}".format(ret))
        else:
            logging.error("push data to YOLO postprocess FAIL, ret: {}".format(ret))
            time.sleep(0.001)
        logging.info("YOLO postprocess done, time use: {:.2f}s".format(time.time() - start_time))

        
        # 2 得到后处理的一张图的结果，并做crop
        crop_time = time.time()
        res = []
        for _ in range(batch_size):
            objs, channel_idx, image_idx = self.yolov5_post_async.get_result_npy() # objs:tuple[left, top, right, bottom, class_id, score] 一张图上的多个检测框

            boxes = []
            for idx in range(len(objs)):
                x1, y1, x2, y2, category_id, score = objs[idx]
                # bbox_dict = [align(x1,2),align(y1,2),align((x2-x1),2),align((y2-y1),2)]
                bbox_dict = [int(x1),int(y1),int(x2-x1),int(y2-y1)]
                boxes.append(bbox_dict)
                logging.debug("channel_idx is {} image_idx is {},len(objs) is{}".format(channel_idx, image_idx, len(objs)))
                logging.debug(bbox_dict)
                logging.info("YOLO postprocess DONE! objs:tuple[left, top, right, bottom, class_id, score] :%s",objs[idx])

            # crop
            croped_list = []
            for box in boxes:
                croped_list.append(self.bmcv.crop(image_dict[(channel_idx,image_idx)],box[0],box[1],box[2],box[3]))

            logging.info("image {} CROP DONE! ".format(image_idx))
            for img in croped_list:
                res.append({(channel_idx, image_idx):img})
            
        logging.info("YOLO crop done, time use: {:.2f}s".format(time.time() - crop_time))
        return res 



    def yolov5_process(self, stop_signal):

        """
        函数功能：
        this funcition is a complete flow for yolov5 process. 

        参数：
        model_path: bmdoel path, must be 4batch 
        ipc_image_pipe_name:str, ipc_dist_pipe_name:str: for init recive ipc,and get data from one ipc
        tpu_id: infer on which tpu
        """

        while not stop_signal.value:
            logging.debug("YOLO process start")
            # 1 get data from ipc and push data to YOLO process

            bmimg_list,channel_id_list,frame_id_list = [], [], []

            for i in range(4):
                try:
                    bmimg, channel_id, frame_id = self.receive_ipc.receiveBMImage()
                except Exception as e:
                    print("An exception occurred:", e)
                    # 打印异常堆栈信息
                    traceback.print_exc()
                    # 可以选择退出程序
                    sys.exit(1)
                bmimg_list.append(bmimg)
                channel_id_list.append(channel_id)
                frame_id_list.append(frame_id)

                logging.debug("YOLO get bmimage from ipc")

            # 2 preprocess and infer 
            output_tensor_map, ost_images, channel_list, imageidx_list, padding_atrr = self.yolov5_pre_infer(self.engine_image_pre_process, channel_id_list, frame_id_list, bmimg_list)
            logging.info("YOLO pre and infer DONE!")

            res = self.yolov5_post(self.yolov5_post_async, self.bmcv, output_tensor_map, ost_images, channel_list, imageidx_list, padding_atrr,batch_size) # res: list[dict{(cid,frame_id):croped_images}]
            logging.info("YOLO post and crop DONE!,len res is %s:",len(res))
        
        
        try:
            logging.info(time.time())
            sys.exit(0)  # 正常退出，退出状态码为0
        except SystemExit as e:
            exit_code = e.code
            print("YOLO Exit code:", exit_code)
            
        
    

