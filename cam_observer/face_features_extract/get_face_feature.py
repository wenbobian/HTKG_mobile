#!/usr/bin/python
# -*- coding:UTF-8 -*-
'''Perform extract face features.'''
# RUC License
# 
# Copyright (c) 2018 Guangzhen Liu,Wenbo Bian
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
import time
import sys
import cv2
import numpy
import mxnet
from glob import glob
from collections import namedtuple

from face_features_extract.Symbol_MobileFace_Identification_V1 import *


class MobileFaceFeatureExtractor(object):
    def __init__(self, model_file, epoch, batch_size, context = mxnet.cpu(), gpu_id = 0):
        self.model_file = model_file
        self.epoch = epoch
        self.batch_size = batch_size
        self.context = context

        network = get_feature_symbol_mobileface_v1()
        self.model = mxnet.mod.Module(symbol = network, context = context)
        self.model.bind(for_training = False, data_shapes=[('data', (self.batch_size, 1, 100, 100))])
        sym, arg_params, aux_params = mxnet.model.load_checkpoint(self.model_file, self.epoch)
        self.model.set_params(arg_params, aux_params)

    def get_face_feature_batch(self, face_batch):
        Batch = namedtuple('Batch', ['data'])
        batch_data = numpy.zeros((self.batch_size, 1, 100, 100))
        face_batch = face_batch.astype(numpy.float32, copy=False)
        face_batch = (face_batch - 127.5)/127.5
        face_num = len(face_batch)
        batch_data[:face_num, 0, :, :] = face_batch
        self.model.forward(Batch([mxnet.nd.array(batch_data)]))
        feature = self.model.get_outputs()[0].asnumpy().copy()
        return feature[:face_num, ...]


if __name__ == "__main__":
    model_file = '../MobileFace_Identification/MobileFace_Identification_V1'
    epoch = 0
    gpu_id = 0
    batch_size = 1
    # context = mxnet.gpu(gpu_id)
    context = mxnet.cpu()
    face_feature_extractor = MobileFaceFeatureExtractor(model_file, epoch, batch_size, context, gpu_id)

    root_path = "../data/LFW-Aligned-100Pair/Aaron_Peirsol/"
    file_names = glob(root_path + '*.*')
    count = 0
    face_batch = []
    for face_one in file_names:
        img = cv2.imread(face_one, 0)
        face_batch.append(img)
        count += 1
        if count % batch_size == 0:
            feature = face_feature_extractor.get_face_feature_batch(numpy.array(face_batch))
            face_batch = []
            print count
            print feature
            print feature.shape


