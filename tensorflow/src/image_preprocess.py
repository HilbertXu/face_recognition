#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Data: 2019/1/1
Author: Xu Yucheng
Abstract: Code for preprocessing image
'''
import tensorflow as tf

def image_preprocess(image, IMAGE_SIZE, IMAGE_SZIE):
    #data_augmentation随机水平翻转图片
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32)/255.0
    return image
