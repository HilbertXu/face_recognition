#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2018/11/23
Author: Xu Yucheng
Abstract: Code for Train model 
'''
import os
import cv2
import sys
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

#user defined modules
from load_data import load_data_main
from model import VGG
from utils import *

TFRECORD_DIR = '/home/kamerider/machine_learning/face_recognition/tensorflow/TFRecords'

def check_dataset():
    train_tfrecord_path = os.path.abspath(os.path.join(TFRECORD_DIR, "train.tfrecords"))
    valid_tfrecord_path = os.path.abspath(os.path.join(TFRECORD_DIR, "valid.tfrecords"))
    if not os.path.exists(train_tfrecord_path) or not os.path.exists(valid_tfrecord_path):
        load_data_main()

'''
@TODO
从生成的.tfrecords文件中读取数据到tf.data.Dataset中
并分batch取出数据
对每一batch的数据进行预处理(图像减去所有图像rgb均值后归一化，转化标签为one-hot码)
研究如何处理检验集的数据
写loss函数， 回调函数等训练时用到的函数
写训练的函数
'''

    
