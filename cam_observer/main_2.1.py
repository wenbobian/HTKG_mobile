#!/usr/bin/python
# -*- coding:UTF-8 -*-
'''Perform the main process of pedestrian in cam video flow, includes pedestrian detect, pedestrian features extract, face detect and face features extract.'''
# RUC License
# 
# Copyright (c) 2018 Wenbo Bian,Guangzhen Liu
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
sys.path.append('/home/nvidia/apache-mxnet-src-1.1.0-incubating/example/image-classification')
from common import find_mxnet, fit

import numpy as np
import cv2
#import socket
#from Tkinter import *
from PIL import Image, ImageDraw, ImageFont

caffe_root = '../caffe/'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import tensorflow as tf
import mxnet as mx
import time

from collections import namedtuple

from face_detect.utils import label_map_util
from face_features_extract.get_face_feature import *

from util.pool import *

NET_FILE = './pedestrian_detect/MobileNetSSD_deploy.prototxt'
CAFFE_MODEL = './pedestrian_detect/MobileNetSSD_deploy10695.caffemodel'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './face_detect/model/frozen_inference_graph_face.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './face_detect/protos/face_label_map.pbtxt'
NUM_CLASSES = 2
MODEL_FLIE = './face_features_extract/MobileFace_Identification_V1'
MX_MODEL_PRE = './pedestrian_features_extract/mobilenet_v2'

'''pedestrian detect settings'''
caffe.set_device(0)
caffe.set_mode_gpu()

try:
    net = caffe.Net(NET_FILE, CAFFE_MODEL, caffe.TEST)
except:
    print('MobileNetSSD_deploy.affemodel does not exist.')
    exit()

CLASSES = ('background', 'person')

'''face detect settings'''
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
with detection_graph.as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=detection_graph, config=config) as sess:
        # face detection process
        def face_detect(candidate_image, face_threshold=.8):
            try:
                image_np = cv2.cvtColor(candidate_image, cv2.COLOR_BGR2RGB)
                img_w = len(candidate_image[0])
                img_h = len(candidate_image)
                # the libopencv_core.so.2.4: cannot open shared objectarray based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
                scores = scores[0].tolist()
                boxes = boxes[0].tolist()
                if max(scores) > face_threshold:
                    max_idx = scores.index(max(scores))
                    box = boxes[max_idx]
                    ymin, xmin, ymax, xmax = box
                    (left, right, top, bottom) = (int(xmin * img_w), int(xmax * img_w), int(ymin * img_h), int(ymax * img_h))
                    # cv2.imshow("SSD", candidate_image[top:bottom, left:right])
                    # scv2.waitKey(1) & 0xff
                        # Exit if ESC pressed
                    face_in_candidate_image = candidate_image[top:bottom, left:right]

                    return face_in_candidate_image
                else:
                    return None
            except:
                return None

'''face feature extract'''
face_feature_extractor = MobileFaceFeatureExtractor(MODEL_FLIE, 0, 1, mxnet.gpu(0), 0)

def face_features_extract(face):
    face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (100, 100), interpolation=cv2.INTER_CUBIC)
    face_batch = []
    face_batch.append(face)
    face_feature = face_feature_extractor.get_face_feature_batch(numpy.array(face_batch))[0]

    return face_feature

'''pedestrian detect process preprocess'''
def preprocess(src):
    if src is None:
	return src
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

'''pedestrian detect process postprocess'''
def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

'''pedestrian detect process detect'''
def detect(origimg):
    candidate_pedestrian_list = []
    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
        if conf[i] > 0.3:
            x_lt = box[i][1]
            y_lt = box[i][0]
            x_rb = box[i][3]
            y_rb = box[i][2]
            candidate_pedestrian_list.append(origimg[x_lt:x_rb, y_lt:y_rb])
            # cv2.imshow("SSD", origimg[x_lt:x_rb, y_lt:y_rb])
            # cv2.waitKey(1) & 0xff
                # Exit if ESC pressed
    return candidate_pedestrian_list

'''pedestrian feature extract'''
Batch = namedtuple('Batch', ['data'])
ctx = mx.gpu(0)
sym, arg_params, aux_params = mx.model.load_checkpoint(MX_MODEL_PRE, 0)
all_layers = sym.get_internals()
fe_sym = all_layers['pool6_output']
fe_mod = mx.mod.Module(symbol=fe_sym, context=ctx, label_names=None)
fe_mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
# fe_mod.set_params(arg_params, aux_params)
fe_mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

def pedestrian_features_extract(pedestrian):
    pedestrian_feature = []
    pedestrian = mx.nd.array(pedestrian)
    img = mx.image.imresize(pedestrian, 224, 224) # resize
    img = img.transpose((2, 0, 1)) # Channel first
    img = img.expand_dims(axis=0) # batchify
    fe_mod.forward(Batch([img]))
    features = fe_mod.get_outputs()[0]
    for i in xrange(len(features.asnumpy()[0])):
        pedestrian_feature.append(features.asnumpy()[0][i][0][0])
    return pedestrian_feature



if __name__ == '__main__':
    # video_capture = cv2.VideoCapture('pedestrian.flv')
    #gst_str = ("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)800, height=(int)600, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=6 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    #video_capture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    video_capture = cv2.VideoCapture(1)
    RATIO = 25
    timer = 0
    pool = Pool()
    cnt = 0
    c_cnt = 0
    in_time = []
    out_time = []
    temp = []
    temp1 = []
    in_1minute_person = 0
    out_1minute_person = 0
   
    
    # address = ('192.168.3.36', 10000)# this computer
    # address = ('192.168.43.123', 10000)
    # readdr = ('192.168.43.44', 9999)
    # readdr = ('192.168.3.34', 9999)# out computer
    #s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #s.setblocking(False)
    #s.bind(address)
    ret, frame = video_capture.read()
    
    while ret:
	# add:opencv.show in and out person
	frame1 = cv2.imread('timg.jpeg')
	img_PIL = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
	font = ImageFont.truetype('NotoSansCJK-Medium.ttc', 80)
	fillColor = (0,0,0)
	position = (150,100)
	position1 = (150,200)
	position2 = (150,300)
	position3 = (150,400)	
	strname =  '进门人数:'
	strname1 = '1分钟进门人数:'
	strname2 = '出门人数:'
	strname3 = '1分钟出门人数:'
	if not isinstance(strname, unicode):
	    strname = strname.decode('utf8')
	if not isinstance(strname1, unicode):
	    strname1 = strname1.decode('utf8')
	if not isinstance(strname2, unicode):
	    strname2 = strname2.decode('utf8')
	if not isinstance(strname3, unicode):
	    strname3 = strname3.decode('utf8')
	draw = ImageDraw.Draw(img_PIL)
	draw.text(position, strname + str(cnt), font = font, fill=fillColor)
	draw.text(position1, strname1 + str(in_1minute_person), font = font, fill=fillColor)
	draw.text(position2, strname2 + str(c_cnt), font = font, fill=fillColor)
	draw.text(position3, strname3 + str(out_1minute_person), font = font, fill=fillColor)	
	frame1 = cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)

	
	'''
	cv2.putText(frame,"in_per_num:" + str(cnt),(450,30),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),2)
	cv2.putText(frame,"out_per_num:" + str(c_cnt),(450,60),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),2)        
	cv2.putText(frame,"in_1min_num:" + str(in_1minute_person),(450,90),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),2)
	cv2.putText(frame,"out_1min_num:" + str(out_1minute_person),(450,120),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),2)
	'''
	'''        
	root = Tk()
        root.title("人数统计")
        root.geometry('500x400')       
	Label(root, text = '通过入口人数:' + str(cnt).decode('gbk').encode('utf8'),font=('Arial',35)).pack()
	Label(root, text = '通过出口人数:' + str(c_cnt).decode('gbk').encode('utf8'),font=('Arial',35)).pack(side = TOP)
	Label(root, text = '1分钟通过入口人数:' + str(in_1minute_person).decode('gbk').encode('utf8'),font=('Arial',35)).pack(side = TOP)
	Label(root, text = '1分钟通过出口人数:' +str(out_1minute_person).decode('gbk').encode('utf8'),font=('Arial',35)).pack(side = TOP)
	'''	
	cv2.imshow('Demo1', frame1)
	cv2.imshow('Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # ret, frame = video_capture.read()
        
        s_t = time.time()
        # pedestrian_and_face_features belongs to the current frame
        pedestrian_and_face_features = []
	pedestrian_feature = []
	pedestrian_features = []
        # get the candidate pedestrians in the current frame
        candidate_pedestrian_list = detect(frame)
        # print len(candidate_pedestrian_list)
        for idx in xrange(len(candidate_pedestrian_list)):  
	    face = face_detect(candidate_pedestrian_list[idx]) #face,return face_in_canditate_image
	    # pedestrian_feature = pedestrian_features_extract(candidate_pedestrian_list[idx])
            if face is not None:
                # get pedestrian feature
                pedestrian_feature = pedestrian_features_extract(candidate_pedestrian_list[idx])
                # get face feature
                face_feature = face_features_extract(face).tolist()
                # merge pedestrian feature and face feature
                pedestrian_and_face_feature = pedestrian_feature + face_feature
                # pedestrian_and_face_features.append(pedestrian_and_face_feature)
		if pool.insert_item(pedestrian_and_face_feature, 700):
		    print 'catch 1 person'
                    cnt += 1
		    print cnt
		    # build mkdir according to time
		    year = time.strftime('%Y',time.localtime(time.time()))
		    month = time.strftime('%m',time.localtime(time.time()))
		    day = time.strftime('%d',time.localtime(time.time()))
		    fileYear = './frame_picture/' + year
		    fileMonth=fileYear+'/'+month
     		    fileDay=fileMonth+'/'+day

		    if not os.path.exists(fileYear):
    		       os.mkdir(fileYear)
    		       os.mkdir(fileMonth)
    		       os.mkdir(fileDay)
		    else:
    		        if not os.path.exists(fileMonth):
        		    os.mkdir(fileMonth)
        		    os.mkdir(fileDay)
    		        else:
        		    if not os.path.exists(fileDay): 
            	 	        os.mkdir(fileDay)
		    fileDir = fileDay + '/'
		    # add:save frame by time
		    cv2.imwrite(fileDir +'%s.jpg'%(str(time.strftime('%Y-%m-%d %H:%M:%S'))), frame)	
		    in_time.append(time.time())
 		    temp = in_time
		else:
		    continue 
	    #elif pedestrian_feature is not None:
	    else:
		pedestrian_features = pedestrian_features_extract(candidate_pedestrian_list[idx])
		if pool.insert_item(pedestrian_features, 700):
		    print 'catch 1 person'
                    c_cnt += 1
		    print c_cnt
		    # build mkdir according to time
		    year = time.strftime('%Y',time.localtime(time.time()))
		    month = time.strftime('%m',time.localtime(time.time()))
		    day = time.strftime('%d',time.localtime(time.time()))
		    fileYear = './out_frame_picture/' + year
		    fileMonth=fileYear+'/'+month
     		    fileDay=fileMonth+'/'+day

		    if not os.path.exists(fileYear):
    		       os.mkdir(fileYear)
    		       os.mkdir(fileMonth)
    		       os.mkdir(fileDay)
		    else:
    		        if not os.path.exists(fileMonth):
        		    os.mkdir(fileMonth)
        		    os.mkdir(fileDay)
    		        else:
        		    if not os.path.exists(fileDay): 
            	 	        os.mkdir(fileDay)
		    fileDir = fileDay + '/'
		    # add:save frame by time
		    cv2.imwrite(fileDir +'%s.jpg'%(str(time.strftime('%Y-%m-%d %H:%M:%S'))), frame)	
		    out_time.append(time.time())
 		    temp1 = out_time
		else:
		    continue 

	out_1minute_person = 0
	'''
	for item in pedestrian_features:
	    if pool.insert_item(item, 700):
	'''
	  
	for i in range(len(temp1)):
	    if temp1[i] >= time.time() - 60:
		out_1minute_person +=1
			
	# print 'out_1minute_person:', out_1minute_person		
	in_1minute_person = 0
	'''
        for item in pedestrian_and_face_features:
            if pool.insert_item(item, 700):
                print 'catch 1 person'
                cnt += 1
		print cnt
		# build mkdir according to time
		year = time.strftime('%Y',time.localtime(time.time()))
		month = time.strftime('%m',time.localtime(time.time()))
		day = time.strftime('%d',time.localtime(time.time()))
		fileYear = './frame_picture/' + year
		fileMonth=fileYear+'/'+month
     		fileDay=fileMonth+'/'+day

		if not os.path.exists(fileYear):
    		    os.mkdir(fileYear)
    		    os.mkdir(fileMonth)
    		    os.mkdir(fileDay)
		else:
    		   if not os.path.exists(fileMonth):
        		os.mkdir(fileMonth)
        		os.mkdir(fileDay)			
    		   else:
        		if not os.path.exists(fileDay): 
            	 	    os.mkdir(fileDay)			    
		fileDir = fileDay + '/'
		# add:save frame by time
		cv2.imwrite(fileDir +'%s.jpg'%(str(time.strftime('%Y-%m-%d %H:%M:%S'))), frame)
		# add:array,time,person number,can see person number past 1 hour
		in_time.append(time.time())
 		temp = in_time		
		print temp	        
            else:
                continue
	'''
	for i in range(len(temp)):
	    if temp[i] >= time.time() - 60:
		in_1minute_person +=1			
	# print 'in_1minute_person:', in_1minute_person
	# a[i] = m //i is person id, m is appear time


	# server(cnt)
        e_t = time.time()
        timer = 1
        pool.refresh(time_threshold=30)
        # print e_t - s_t

        ret, frame = video_capture.read()
'''        
        try:
            data, addr = s.recvfrom(2048)
            c_cnt = int(data.decode())	
	    # add:array,time,person number,can see person number past 1 hour
	    out_time.append(time.localtime(time.time()).tm_min*60+time.localtime(time.time()).tm_sec)
 	    temp = out_time
	    out_1minute_person = 0
	    for i in range(len(temp)):
	    	if temp[i] >= time.localtime(time.time()).tm_min*60 + time.localtime(time.time()).tm_sec - 60:
		    out_1minute_person +=1
	    print 'out_1minute_person:', out_1minute_person
            #
            #print c_cnt
            #print cnt
            print cnt - c_cnt
	    
        except:
            #print c_cnt
            #print cnt
            print cnt - c_cnt
'''   
	# root.mainloop()
# s.close()








