#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2018/11/23
Author: Xu Yucheng
Abstract: A test code for tf.data.Dataset
'''

import tensorflow as tf
import numpy as np 
from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

if __name__ == '__main__':
    image_path = ['/home/kamerider/Documents/DataBase/1610763/10000.png','/home/kamerider/Documents/DataBase/1610763/10000.png']
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
