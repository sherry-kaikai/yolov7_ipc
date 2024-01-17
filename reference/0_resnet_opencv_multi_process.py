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
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.input_shapes = [self.net.get_input_shape(self.graph_name, name) for name in self.input_names]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_shapes = [self.net.get_output_shape(self.graph_name, name) for name in self.output_names]
        logging.debug("load {} success!".format(args.bmodel))
        logging.debug(str(("graph_name: {}, input_names & input_shapes: ".format(self.graph_name), self.input_names, self.input_shapes)))
        logging.debug(str(("graph_name: {}, output_names & output_shapes: ".format(self.graph_name), self.output_names, self.output_shapes)))
        self.input_name = self.input_names[0]
        self.input_shape = self.input_shapes[0]

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

        self.handle = self.net.get_handle()
        self.bmcv = sail.Bmcv(self.handle)
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_names[0])
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)

    def preprocess(self, img):
        h, w, _ = img.shape
        if h != self.net_h or w != self.net_w:
            img = cv2.resize(img, (self.net_w, self.net_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img = (img/255-self.mean)/self.std
        img = np.transpose(img, (2, 0, 1))
        return img

    def predict(self, input_img):
        input_data = {self.input_name: input_img}
        # print(input_img.shape)
        outputs = self.net.process(self.graph_name, input_data)
        return list(outputs.values())[0]

    def postprocess(self, outputs):
        res = list()
        outputs_exp = np.exp(outputs)
        outputs = outputs_exp / np.sum(outputs_exp, axis=1)[:,None]
        predictions = np.argmax(outputs, axis = 1)
        for pred, output in zip(predictions, outputs):
            score = output[pred]
            res.append((pred.tolist(),float(score)))
        return res

    def __call__(self, img_list):
        img_num = len(img_list)
        img_input_list = []
        for img in img_list:
            start_time = time.time()
            img = self.preprocess(img)
            self.preprocess_time += time.time() - start_time
            img_input_list.append(img)
        
        if img_num == self.batch_size:
            input_img = np.stack(img_input_list)
            start_time = time.time()
            outputs = self.predict(input_img)
            self.inference_time += time.time() - start_time
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(img_input_list)
            start_time = time.time()
            outputs = self.predict(input_img)[:img_num]
            self.inference_time += time.time() - start_time
        
        start_time = time.time()
        res = self.postprocess(outputs)
        self.postprocess_time += time.time() - start_time

        return res

    def get_time(self):
        return self.dt


'''
send 
解码,并传入numpy
'''
def decode_images(input_path,queue,frame_count):
    # print('---------------sender---------------------')
    frame_id = 0
    if os.path.isfile(input_path): # 视频流
        # print('************************is video************************')
        cap = cv2.VideoCapture(input_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                frame_count.value += frame_id
                queue.put(None)
                break
            frame_id += 1
            queue.put((0,frame_id,frame))


    elif os.path.isdir(input_path):
        for root, dirs, filenames in os.walk(input_path):
            for filename in filenames:
                # print('************************',filename,'************************')
                img_file = os.path.join(root, filename)
                frame = cv2.imread(filename)

                queue.put((0,frame_id,img_file))
                frame_id += 1
        frame_count.value += frame_id
        queue.put(None)


'''
recive
推理,获取numpy
'''
def preprocess_images(resnet,queue):
    # print('---------------recive---------------------')
    resive_end = 0
    while(True):
        img = np.zeros((resnet.batch_size, 3, 224, 224))
        for i in range(resnet.batch_size):
            data = queue.get()

            if data is None:
                print('---------------recive end---------------------')
                resive_end = 1
                break
            cid, frameid, image = data
            img[i] = resnet.preprocess(image)
        if not resive_end:
            outputs = resnet.predict(img)
            res = resnet.postprocess(outputs)
            logging.info("fileid: {}, res: {}".format(frameid, res[i]))
        else:
            break


def main(args):
    frame_count= multiprocessing.Value("d",0) # 声明一个多进程共享数值
    
    resnet = Resnet(args)
    input_video = args.input
    queues = [multiprocessing.Queue(maxsize=args.queue_size) for _ in range(args.process_num)]

    send_processes = [multiprocessing.Process(target=decode_images, args=(input_video,queue,frame_count)) for queue in queues]
    receive_processes = [multiprocessing.Process(target=preprocess_images, args=(resnet,queue)) for queue in queues]

    start_time = time.time()

    # 启动进程
    for send_process in send_processes:
        send_process.start()
    for receive_process in receive_processes:
        receive_process.start()

    # 等待进程结束

    for send_process in send_processes:
        send_process.join()
    for receive_process in receive_processes:
        receive_process.join()

    end_time = time.time() - start_time
    # calculate speed  
    print(frame_count.value)
    print("total_time(s): {:.2f}".format(end_time))
    print("total_fps: ",(frame_count.value/end_time))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    # parser.add_argument('--input', type=str, default='../../data/images/face_data_test', help='path of input, must be image directory')
    parser.add_argument('--input', type=str, default='../data/monitor.mp4', help='path of input, must be image directory')
    parser.add_argument('--process_num', type=int, default=1, help='path of input, must be image directory')
    parser.add_argument('--queue_size', type=int, default=8, help='path of input, must be image directory')
    parser.add_argument('--bmodel', type=str, default='../data/BM1684/resnet50_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    logging.basicConfig(filename='example.log', level=logging.INFO)
    main(args)
