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
from load_data import load_data_main RGB_MEAN
from model import VGG
from utils import *

TFRECORD_DIR = '/home/kamerider/machine_learning/face_recognition/tensorflow/TFRecords'

train_data_size=0
valid_data_size=0
class_num = 0

def check_dataset():
	train_tfrecord_path = os.path.abspath(os.path.join(TFRECORD_DIR, "train.tfrecords"))
	valid_tfrecord_path = os.path.abspath(os.path.join(TFRECORD_DIR, "valid.tfrecords"))
	if not os.path.exists(train_tfrecord_path) or not os.path.exists(valid_tfrecord_path):
		load_data_main()
		check_dataset()
	else:
		with open(os.path.abspath(os.path.join(TFRECORD_DIR, "train.txt")), 'r') as f:
			global train_data_size
			train_data_size = int(f.readline())
			global class_num
			class_num = int(f.readline())
		f.close()

		with open(os.path.abspath(os.path.join(TFRECORD_DIR, "valid.txt")), 'r') as f:
			global valid_data_size
			valid_data_size = int(f.readline())
		f.close()
	return train_data_size, valid_data_size

def _parse_tfrecord(example_proto):
	#定义解析用的字典
	dics = {}
	dics['image_raw'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
	dics['label'] = tf.FixedLenFeature(shape=[],dtype=tf.int64)

	#调用接口解析一行样本
	parsed_data = tf.parse_single_example(serialized=example_prote, features=dics)
	image_raw = tf.decode_raw(parsed_data['image_raw'], out_type=tf.uint8)
	image_raw = tf.reshape(image_raw, shape=[64,64])
	
	#对图片数据进行归一化处理
	#这里对图像数据做归一化，是关键，没有这一步，精度不收敛，为0.1左右
	#先减去之前load_data时得到的所有训练图像RGB均值
	RGB_MEAN = tf.constant(RGB_MEAN, dtype=tf.float32)
	image_centered = tf.subtract(image_raw, RGB_MEAN)

	#对像素值进行归一化
	image = tf.cast(image_centered, tf.float32)*(1./255)

	#处理图像对应的标签，将其转化为one-hot code
	label = parsed_data['label']
	label = tf.cast(label, tf.int32)
	label = tf.one_hot(label, depth=class_num, on_value=1)

	return image, label




'''
@TODO
从生成的.tfrecords文件中读取数据到tf.data.Dataset中
并分batch取出数据
对每一batch的数据进行预处理(图像减去所有图像rgb均值后归一化，转化标签为one-hot码)
研究如何处理检验集的数据
写loss函数， 回调函数等训练时用到的函数
写训练的函数
'''
def read_train_tfrecord(filename):
	full_path = os.path.abspath(os.path.join(TFRECORD_DIR, filename))
	train_dataset = tf.data.TFRecordDataset(filename=['train.tfrecords'])
	train_dataset = train_dataset.map()

if __name__ == '__main__':
	train_data_size, valid_data_size = check_dataset()
	print (train_data_size, valid_data_size)
	print (class_num)	
	read_tfrecord("train.tfrecords")



    
