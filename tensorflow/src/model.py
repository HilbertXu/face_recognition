# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
Date: 2018/12/29
Author: Xu Yucheng
Abstract: Code for build VGG-16 model
'''

import tensorflow as tf
from ops import *


#input_op: 每一个batch输入网络的图像数据
#keep_prob: drop_out 层中保留神经元链接的比例，在测试的时候赋值为1 
def VGG(input_op, keep_prob=0.5, class_num=62):
    #通过依次采用多个3x3的卷积核，模仿出更大的感受野的效果
    p=[]
    #第一层卷积层用作输入层
    #此后经过卷积核逐渐增多的5次卷积
    conv1_1 = conv2d(input_op, name='conv1_1', output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
    conv1_2 = conv2d(conv1_1, name='conv1_2', output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
    mpool1 = mpool_layer(conv1_2, name='mpool1', k_h=2, k_w=2, d_h=2, d_w=2)
    conv1_drop = dropout(mpool1, name='conv1_drop', keep_prob=0.75)

    conv2_1 = conv2d(conv1_drop, name='conv2_1', output_dim=128, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
    conv2_2 = conv2d(conv2_1, name='conv2_2', output_dim=128, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
    mpool2 = mpool_layer(conv2_2, name='mpool2', k_h=2, k_w=2, d_h=2, d_w=2)
    conv2_drop = dropout(mpool2, name='conv2_drop', keep_prob=0.75)

    conv3_1 = conv2d(conv2_drop, name='conv3_1', output_dim=256, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
    conv3_2 = conv2d(conv3_1, name='conv3_2', output_dim=256, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
    conv3_3 = conv2d(conv3_2, name='conv3_3', output_dim=256, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
    mpool3 = mpool_layer(conv3_3, name='mpool3', k_h=2, k_w=2, d_h=2, d_w=2)
    conv3_drop = dropout(mpool3, name='conv3_drop', keep_prob=0.7)

    conv4_1 = conv2d(conv3_drop, name='conv4_1', output_dim=512, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
    conv4_2 = conv2d(conv4_1, name='conv4_2', output_dim=512, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
    conv4_3 = conv2d(conv4_2, name='conv4_3', output_dim=512, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
    mpool4 = mpool_layer(conv4_3, name='mpool4', k_h=2, k_w=2, d_h=2, d_w=2)
    conv4_drop = dropout(mpool4, name='conv4_drop', keep_prob=0.7)

    conv5_1 = conv2d(conv4_drop, name='conv5_1', output_dim=512, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
    conv5_2 = conv2d(conv5_1, name='conv5_2', output_dim=512, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
    conv5_3 = conv2d(conv5_2, name='conv5_3', output_dim=512, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
    mpool5 = mpool_layer(conv5_3, name='mpool5', k_h=2, k_w=2, d_h=2, d_w=2)
    conv5_drop = dropout(mpool5, name='conv5_drop', keep_prob=0.7)

    flatten = flatten_layer(conv5_drop, name='flatten')

    fc6 = fc_layer(flatten, name='fc6', output_shape=4096, p=p)
    fc6_drop = dropout(fc6, name='fc6_drop', keep_prob=keep_prob)

    fc7 = fc_layer(fc6_drop, name='fc7', output_shape=4096, p=p)
    fc7_drop = dropout(fc7, name='fc7_drop', keep_prob=keep_prob)
    
    #最后一层全连接层需要将之前层的输出激活62个类别对应的62个神经元
    #故fc8的output_shape=62
    fc8 = fc_layer(fc7_drop, name='fc8', output_shape=class_num, p=p)
    softmax = softmax_layer(fc8, name='softmax')
    
    predicitions = tf.argmax(softmax, 1)

    return predicitions, softmax, p

def save_model(sess, save_path):
    with tf.name_scope("saver"):
        saver =  tf.train.Saver()
        saver.save(sess, save_model)