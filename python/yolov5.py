from sophon import sail
import time
import logging
import numpy as np
logging.getLogger()   
import sys
import traceback
def align(num, align_to):
    align_to = align_to - 1
    return  (num + align_to) & (~align_to)

def yolov5_pre_infer(engine_image_pre_process, channel_id_list, frame_id_list, bmi_list):
    '''
    参数：
    engine_image_pre_process: This is a sail method that has been initialized
    '''
    time_start = time.time()
    
    for i in range(len(bmi_list)):
        bmi = bmi_list[i]
        logging.info("pushdata start, bmimg w is {}, h is {} ".format(bmi.width(),bmi.height()) )
        ret = engine_image_pre_process.PushImage(channel_id_list[i], frame_id_list[i], bmi)
        while(ret != 0): # 如果push失败，等待
            logging.info("Porcess[{}]:[{}]PreProcessAndInference Thread Full, sleep: 10ms!".format(channel_id_list[i],frame_id_list[i]))
            time.sleep(0.01)

    logging.info("pushdata exit, time use: {:.2f}s".format(time.time()-time_start))

    res = engine_image_pre_process.GetBatchData(True)
    logging.info("YOLO pre and process done, time use: {:.2f}s".format(time.time()-time_start))
    return  res # output_tensor_map, ost_images, channel_list, imageidx_list, padding_atrr



def yolov5_post(yolov5_post_async,bmcv, output_tensor_map:list, ost_images:list, channel_list:list, imageidx_list:list, padding_atrr:list, batch_size:int, dete_threshold = 0.65, nms_threshold = 0.6, tpu_id = 0):
    '''
    功能：
        调用sail.algo_yolov5_post_cpu_opt_async 进行批后处理，一次处理一个batch的图片，并将小图从原图中crop出来。

    输入：
        yolo_output_shape: bmodel的输出shape，用于后处理
        output_tensor_map: engine_image_pre_process.GetBatchData(True)输出的第一个数据，即为推理结果 
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
        padding_atrr:list[list[int]]:填充图像的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度。
        batch_size:int: 输入batch，确保输入的output_tensor_map的长度是bmodel的一个batch
        dete_threshold = 0.01:
        nms_threshold = 0.6:
        tpu_id = 0:

    返回值：list[dict{(cid,frame_id):croped_images}]    (？这里要返回原图吗？)
        

    注意：
        需要cid、imageid 来一一对应做crop的原图。

    '''
    start_time = time.time()

    # 0 处理数据
    # 获取原图宽高
    width_list = []
    height_list= []
    image_dict = {}
    for i in range(len(ost_images)):
        width_list.append(ost_images[i].width())
        height_list.append(ost_images[i].height())
        image_dict[(channel_list[i],imageidx_list[i])] = ost_images[i]

    dete_thresholds = np.ones(len(channel_list),dtype=np.float32)
    nms_thresholds = np.ones(len(channel_list),dtype=np.float32)
    dete_thresholds = dete_threshold*dete_thresholds
    nms_thresholds = nms_threshold*nms_thresholds

    logging.info("YOLO postprocess init done, time use: {:.2f}s".format(time.time() - start_time))
    # 1 后处理
    ret = yolov5_post_async.push_data(channel_list, imageidx_list, output_tensor_map, dete_thresholds, nms_thresholds, width_list, height_list, padding_atrr)
    if ret == 0:
        logging.debug("push data to YOLO postprocess SUCCESS, ret: {}".format(ret))
    else:
        logging.error("push data to YOLO postprocess FAIL, ret: {}".format(ret))
        time.sleep(0.01)

    logging.info("YOLO postprocess done, time use: {:.2f}s".format(time.time() - start_time))

    crop_time = time.time()
    # 2 得到后处理的一张图的结果，并做crop
    res = []
    for _ in range(batch_size):
        objs, channel_idx, image_idx = yolov5_post_async.get_result_npy() # objs:tuple[left, top, right, bottom, class_id, score] 一张图上的多个检测框。

        boxes = []
        for idx in range(len(objs)):
            x1, y1, x2, y2, category_id, score = objs[idx]
            bbox_dict = [align(x1,32),align(y1,2),align((x2-x1),16),align((y2-y1),4)]
            boxes.append(bbox_dict)
            logging.debug("channel_idx is {} image_idx is {},len(objs) is{}".format(channel_idx, image_idx, len(objs)))
            logging.debug(bbox_dict)
            logging.info("YOLO postprocess DONE! objs:tuple[left, top, right, bottom, class_id, score] :%s",objs[idx])
        '''
        test 1  段错误
        '''
        # croped_list = bmcv.crop(image_dict[(channel_idx,image_idx)],boxes)

        '''
        test2 一次一次的crop
        '''
        croped_list = []
        for box in boxes:
            croped_list.append(bmcv.crop(image_dict[(channel_idx,image_idx)],box[0],box[1],box[2],box[3]))

        logging.info("image {} CROP DONE! ".format(image_idx))
        for img in croped_list:
            res.append({(channel_idx, image_idx):img})
        
    logging.info("YOLO crop done, time use: {:.2f}s".format(time.time() - crop_time))
    return res 



def yolov5_process(stop_signal, ipc_recive_queue_len:int, model_path:str, ipc_image_pipe_name:str, ipc_dist_pipe_name:str, tpu_id:int=0):

    """
    函数功能：
    this funcition is a complete flow for yolov5 process. 

    参数：
    model_path: bmdoel path, must be 4batch 
    ipc_image_pipe_name:str, ipc_dist_pipe_name:str: for init recive ipc,and get data from one ipc
    tpu_id: infer on which tpu
    """
    while True:
        time.sleep(0.01)
        logging.debug("!!!!!!!!!!!!!!!!!!!!!!i am alive!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        

    # 0 init
    resize_type = sail.sail_resize_type.BM_PADDING_TPU_LINEAR # may rights
    # resize_type = sail.sail_resize_type.BM_RESIZE_TPU_NEAREST # wrong
    alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)

    engine_image_pre_process = sail.EngineImagePreProcess(model_path, tpu_id, use_mat_output=0) # use_mat_output 是否使用OpenCV的Mat作为图片的输出，默认为False，不使用。
    engine_image_pre_process.InitImagePreProcess(resize_type, True, 10, 10)
    engine_image_pre_process.SetPaddingAtrr()
    engine_image_pre_process.SetConvertAtrr(alpha_beta)
    net_w = engine_image_pre_process.get_input_width()
    net_h = engine_image_pre_process.get_input_height()


    output_names = engine_image_pre_process.get_output_names()
    batch_size = engine_image_pre_process.get_output_shape(output_names[0])[0]
    output_shapes = [engine_image_pre_process.get_output_shape(i) for i in output_names]
    logging.debug('YOLO init engine_image_pre_process output_shapes %s',output_shapes)
    receive_ipc = sail.IPC(False, ipc_image_pipe_name, ipc_dist_pipe_name, True, ipc_recive_queue_len) # ,10
    logging.debug('YOLO RECEIVE IPC INIT')

    while not stop_signal.value:
        logging.debug("YOLO process start")
        # 1 get data from ipc and push data to YOLO process

        bmimg_list,channel_id_list,frame_id_list = [], [], []

        for i in range(4):
            try:
                bmimg, channel_id, frame_id = receive_ipc.receiveBMImage()
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
        output_tensor_map, ost_images, channel_list, imageidx_list, padding_atrr = yolov5_pre_infer(engine_image_pre_process, channel_id_list, frame_id_list, bmimg_list)
        logging.info("YOLO pre and infer DONE!")

        # 3 post process  
        yolov5_post_async = sail.algo_yolov5_post_cpu_opt_async([output_shapes[0],output_shapes[1],output_shapes[2]],640,640,10)
        bmcv = sail.Bmcv(sail.Handle(0))
        res = yolov5_post(yolov5_post_async, bmcv, output_tensor_map, ost_images, channel_list, imageidx_list, padding_atrr,batch_size) # res: list[dict{(cid,frame_id):croped_images}]
        logging.info("YOLO post and crop DONE!,len res is %s:",len(res))
    
    
    try:
        logging.info(time.time())
        sys.exit(0)  # 正常退出，退出状态码为0
    except SystemExit as e:
        exit_code = e.code
        print("YOLO Exit code:", exit_code)
            
        
    

