import yolov5
import sophon.sail as sail
import cv2
import logging
logging.getLogger()   
logging.basicConfig(filename= f'test.log',  filemode='w',level=logging.DEBUG)

if __name__ == '__main__':
    input_path = '../data/licenseplate.mp4'
    handle = sail.Handle(0)
    decoder = sail.Decoder(input_path)
    frame_id = 0
    frame_id_list = []

    resize_type = sail.sail_resize_type.BM_RESIZE_TPU_NEAREST
    alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)

    engine_image_pre_process = sail.EngineImagePreProcess('../models/yolov5s-licensePLate/BM1684X/yolov5s_v6.1_license_3output_int8_4b.bmodel', 0, use_mat_output=0) # use_mat_output 是否使用OpenCV的Mat作为图片的输出，默认为False，不使用。
    engine_image_pre_process.InitImagePreProcess(resize_type, True, 10, 10)
    engine_image_pre_process.SetPaddingAtrr()
    engine_image_pre_process.SetConvertAtrr(alpha_beta)
    net_w = engine_image_pre_process.get_input_width()
    net_h = engine_image_pre_process.get_input_height()


    output_names = engine_image_pre_process.get_output_names()
    batch_size = engine_image_pre_process.get_output_shape(output_names[0])[0]
    output_shapes = [engine_image_pre_process.get_output_shape(i) for i in output_names]

    times = 0
    while True:
        
        print(times)
        bmi_list = []
        for i in range(4):
            
            bmi = decoder.read(handle)
            bmi_list.append(bmi)
            frame_id_list.append(frame_id)
            frame_id += 1
            
        
        print(bmi.width(),bmi.height())
        
        logging.debug("YOLO process initial")


        logging.debug("YOLO process start")
        '''
        这里是否需要凑4batch的输入图像，还是直接push？在sail.EngineImagePreProcess里面会自动批处理？

        先不在外面凑batch了，直接push吧
        '''
        '''
        不凑4batch不行!?
        '''

        # 2 preprocess and infer 
        output_tensor_map, ost_images, channel_list, imageidx_list, padding_atrr = yolov5.yolov5_pre_infer(engine_image_pre_process, [0,0,0,0], frame_id_list, bmi_list)
        logging.info("yolo pre and infer DONE!")

        # 3 post process  
        yolov5_post_async = sail.algo_yolov5_post_cpu_opt_async([output_shapes[0],output_shapes[1],output_shapes[2]],640,640,10)
        bmcv = sail.Bmcv(sail.Handle(0))

        res = yolov5.yolov5_post(yolov5_post_async,bmcv,output_tensor_map, ost_images, channel_list, imageidx_list, padding_atrr,batch_size) # res: list[dict{(cid,frame_id):croped_images}]
        logging.info("yolo post and crop DONE!,len res is %s:",len(res))
        
        bmi_list.clear()
        times += 1