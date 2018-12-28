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

if __name__ == '__main__':
    image_path = '/home/kamerider/Documents/DataBase/1610763/10000.png'
    label_path = '/home/kamerider/Documents/DataBase/1611260'
    label = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    data = np.random.uniform(size=(12,3))
    dataset = Dataset.from_tensor_slices((label, data))
    
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            while True:
                print(sess.run(one_element))
        except tf.errors.OutOfRangeError:
            print("end!")
    
    '''
    with tf.Session() as sess:
        for i in range(3):
            print (sess.run(one_element))
    '''
    VGG_MEAN = tf.constant([123.68,116.779,103.939], dtype=tf.float32)
    MEAN = tf.constant([111], dtype=tf.float32)
    MEAN_1 = tf.constant([222,222,222], dtype=tf.float32)
    img_string = tf.read_file(image_path)
    img_decoded = tf.image.decode_png(img_string, channels=3)

    with tf.Session() as sess:
        img_resized=tf.image.resize_images(img_decoded, [227,227])
        img_centered = tf.subtract(img_resized, VGG_MEAN)
        print (sess.run(img_centered))
        print (img_centered)
        print (VGG_MEAN)
        print (MEAN)
        #print (MEAN_1)
        print (img_resized.eval())
