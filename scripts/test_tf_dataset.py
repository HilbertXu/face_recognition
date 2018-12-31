#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2018/11/23
Author: Xu Yucheng
Abstract: A test code for tf.data.Dataset
'''
import os
import tensorflow as tf
import numpy as np 
from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from vgg_preprocessing import *

TFRECORD_DIR = '/home/kamerider/machine_learning/face_recognition/tensorflow/TFRecords'

def simple_test():
    image_path = ['/home/kamerider/Documents/DataBase/1610763/10000.png','/home/kamerider/Documents/DataBase/1610763/10000.png','/home/kamerider/Documents/DataBase/1610763/10000.png','/home/kamerider/Documents/DataBase/1610763/10000.png','/home/kamerider/Documents/DataBase/1610763/10000.png']
    label = np.array([1,2])
    data = np.random.uniform(size=(12,3))

    image_path = convert_to_tensor(image_path, dtype=dtypes.string)
    label = convert_to_tensor(label, dtype=dtypes.int32)

    dataset = Dataset.from_tensor_slices((image_path, label))

    
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            while True:
                result = sess.run(one_element)
                #print(result[0])
                image_string = tf.read_file(result[0])
                image_decode = tf.image.decode_png(image_string, channels=3)
                image_resize = tf.image.resize_images(image_decode,[64, 64])
                print (image_resize)
        except tf.errors.OutOfRangeError:
            print("end!")
    
    '''
    with tf.Session() as sess:
        for i in range(3):
            print (sess.run(one_element))
    '''

    '''
    VGG_MEAN = tf.constant([123.68,116.779,103.939], dtype=tf.float32)
    MEAN = tf.constant([111], dtype=tf.float32)
    MEAN_1 = tf.constant([222,222,222], dtype=tf.float32)
    img_string = tf.read_file(image_path)
    img_decoded = tf.image.decode_png(img_string, channels=3)

    with tf.Session() as sess:
        img_resized=tf.image.resize_images(img_decoded, [227,227])
        img_centered = tf.subtract(img_resized, VGG_MEAN)
        #print (sess.run(img_centered))
        #print (img_centered)
        #print (VGG_MEAN)
        #print (MEAN)
        #print (MEAN_1)
        #print (img_resized.eval())
    '''
def _parse_tfrecord(example_proto):
	#定义解析用的字典
	features = {
		'image_raw': tf.FixedLenFeature(shape=[],dtype=tf.string),
		'label': tf.FixedLenFeature(shape=[],dtype=tf.int64),
		'height': tf.FixedLenFeature(shape=[],dtype=tf.int64),
		'width': tf.FixedLenFeature(shape=[],dtype=tf.int64),
		'channels': tf.FixedLenFeature(shape=[],dtype=tf.int64)
	}
	#调用接口解析一行样本
	parsed_data = tf.parse_single_example(serialized=example_proto, features=features)
	#获取图像基本属性
	label, height, width, channels = parsed_data['label'], parsed_data['height'], parsed_data['width'], parsed_data['channels']
	#获取原始二进制图像，并进行转化
	image_decoded = tf.decode_raw(parsed_data['image_raw'], tf.uint8)
	'''
	读取 TFRecord 文件过程中，解析 Example Protobuf 文件时，
	decode_raw 得到的数据(如 image raw data) 要通过 reshape 操作恢复 shape，
	而 shape 参数也是从 TFRecord 文件中获取时，要加 tf.stack 操作: image = tf.reshape(image, tf.stack([height, width, channels]))
	'''
	image_decoded = tf.reshape(image_decoded, tf.stack([height, width, channels]))
	#处理图像对应的标签，将其转化为one-hot code
	label = tf.cast(label, tf.int32)
	label = tf.one_hot(label, depth=62, on_value=1)

	return image_decoded, label

def data_generator(iterator):
    image, label = iterator.get_next()
    return image,label


def test_tfrecord():
    valid_file_path = os.path.abspath(os.path.join(TFRECORD_DIR, "train.tfrecords"))
    dataset_valid = tf.data.TFRecordDataset(valid_file_path)
    dataset_valid = dataset_valid.map(_parse_tfrecord)
    dataset_valid = dataset_valid.map(lambda image, label: (preprocess_image(image, 64, 64, is_training=False), label), num_parallel_calls=2)
    dataset_valid = dataset_valid.batch(64)
    dataset_valid = dataset_valid.prefetch(5)

    valid_iterator = dataset_valid.make_initializable_iterator()
    valid_iterator = dataset_valid.make_one_shot_iterator()
    num = 0
    with tf.Session() as sess:
        try:
            while True:
                #sess.run(tf.global_variable_initializer())
                #if use initializable_iterator() plz uncomment the follwing initializer line
                #sess.run(valid_iterator.initializer)
                #valid_image_batch, valid_label_batch = valid_iterator.get_next()
                valid_image, valid_label = data_generator(valid_iterator)
                valid_image = sess.run(valid_image)
                num+=1
                print("%d batch"%(num))
                print(valid_image.shape) #image_raw
                #print(sess.run(valid_iterator.get_next())[1].shape) #label
        except tf.errors.OutOfRangeError:
            print("end!")

def test_batch():
    test = np.random.randint(0,10,size=(3,4))
    dataset = tf.data.Dataset.from_tensor_slices(test)
    dataset = dataset.repeat(2)
    dataset = dataset.shuffle(3)
    dataset = dataset.batch(2)
    dataset = dataset.prefetch(5)
    iterator = dataset.make_one_shot_iterator()
    for i in range(3):
        with tf.Session() as sess:
            while True:
                try:
                    #sess.run(iterator.initializer)
                    one_batch = sess.run(iterator.get_next())
                    print (one_batch)
                except tf.errors.OutOfRangeError:
                    print("end!")
                    break
        
    



    #使用make_one_shot_interator()可以配合OutOfRangeError来判断是否读取完一次所有的训练数据




            


if __name__ == '__main__':
    test_tfrecord()
    test_batch()