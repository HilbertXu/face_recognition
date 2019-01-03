#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2018/11/23
Author: Xu Yucheng
Abstract: Code for check dataset with tf.read_file
'''

import os
import sys
import shutil
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

DATA_BASE = '/home/kamerider/Documents/DataBase'

image_paths=[]

def find_image(path):
    for File in os.listdir(path):
        full_path = os.path.abspath(os.path.join(path, File))
        if os.path.isdir(full_path):
            find_image(full_path)
        elif File.endswith('.png'):
            image_paths.append(full_path)

def read_image_with_tf(image_paths):

    image_paths = convert_to_tensor(image_paths, dtype=dtypes.string)
    image = tf.data.Dataset.from_tensor_slices(image_paths)

    iterator = image.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            while True:
                #_read_image_with_tf(sess.run(one_element))
                print (sess.run(one_element))
                full_path = sess.run(one_element)
                _read_image_with_tf(full_path)
        except tf.errors.OutOfRangeError:
            print ("end!")


def _read_image_with_tf(filename):
    print ("Reading %s"%(filename))
    img_string = tf.read_file(filename)
    img_decode = tf.image.decode_png(img_string, channels=3)



if __name__ == '__main__':
    find_image(DATA_BASE)
    print (len(image_paths))
    read_image_with_tf(image_paths)
