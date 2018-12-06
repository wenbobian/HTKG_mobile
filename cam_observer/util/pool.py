#!/usr/bin/python
# -*- coding:UTF-8 -*-
'''Perform door pool.'''
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

import numpy as np
import time

class Pool():
    def __init__(self):
        self.pool = {}
    # out = Pool()
    '''insert new item into the pool'''
    def insert_item(self, item, threshold): #item is features,such as [0.162, 0.912, ...]
        idx = self.search_item(item, threshold)
        if idx is -1:
            self.pool[time.time()] = item
            return True
        else:
            return False
    '''refresh the pool'''
    def refresh(self, time_threshold=1):
        this_moment = time.time()
        for key in self.pool.keys():
            if this_moment - key > time_threshold:
                item = self.pool.pop(key)
            else:
                continue
            

    '''check out if the current item is in the pool'''
    def search_item(self, item, threshold):
        max_sim = 0
        index = -1
        for key in self.pool.keys():
            sim = np.linalg.norm(np.array(self.pool[key]) - np.array(item))
            sim = 1 / (1 + sim)
            if sim > max_sim:
                max_sim = sim
                index = key
            else:
                continue
        print '***************************************'
        print max_sim
        if max_sim > threshold:
            self.pool[index] = item
            return index
        else:
            return -1
        
