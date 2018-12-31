#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2018/11/23
Author: Xu Yucheng
Abstract: Code for Create Layers
'''
import tensorflow as tf
from tensorflow.layers import batch_normalization as batch_norm
import numpy as numpy

#网络层的常数偏置
def bias(name, shape, bias_start=0.0, trainable=True):
    dtpye = tf.float32
    var = tf.get_variable(
        name, shape, tf.float32, trainable=trainable,
        initializer=tf.constant_initializer(
            bias_start, dtype=dtpye
        )
    )
    return var

#随机权重
def weight(
    name, shape, stddev=0.03, trainable=True, 
    initializer_weight=tf.random_normal_initializer(stddev=0.03, dtype=tf.float32)
    ):

    dtype=tf.float32
    var = tf.get_variable(
        name, shape, tf.float32, trainable=trainable,
        initializer=initializer_weight
        )
    return var

##Relu激活层
def relu(input_op, name='relu'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return tf.nn.relu(input_op)

#softmax分类输出层
def softmax_layer(intput_op, name='softmax'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return tf.nn.softmax(intput_op)

#Layers
#batch_normalization
'''
公式如下：

y=γ(x-μ)/σ+β

其中x是输入，y是输出，μ是均值，σ是方差，γ和β是缩放（scale）、偏移（offset）系数
'''
def batch_normal(input_op, name='batch_normal', is_train=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        if is_train:
            return batch_norm(
                input_op, decay=0.9, epsilon=1e-5, scale=True,
                is_training=is_train,
                update_collections=None, scope=scope
            )
        else:
            return batch_norm(
                input_op, decay=0.9, epsilon=1e-5, scale=True,
                is_training=is_train, reuse=True,
                update_collections=None, scope=scope
            )

#卷积层
def conv2d(input_op, name, output_dim, k_h, k_w,
            d_h, d_w, p, with_bn=False):
    '''
    tf.nn.conv2d (input, filter, strides, padding, 
                    use_cudnn_on_gpu=None, data_format=None, name=None)

    **input : ** 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。（也可以用其它值，但是具体含义不是很理解）
    **filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
    **strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
    **padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
    **use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true

    '''
    if not with_bn:
        strides = [1, d_h, d_w, 1]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            #见上方注释，此处实际上是使用之前定义的随机权重函数生成output_dim个权重随机的卷积核
            kernel = weight(
                'kernel_w',
                [k_h, k_w, input_op.get_shape()[-1], output_dim]
            ) #input_op.get_shape()[-1]为反向读取input_op的最后一个维度的大小

            conv = tf.nn.conv2d(input_op, kernel, strides=strides, padding='SAME')
            #padding: SAME   用0填充边界，使得每次卷积之后图像尺寸不变，一般需要接一个池化层来对图像降维
            #         VALID  不填充边界，每次卷积后图像尺寸发生改变，改变大小根据卷积核大小而定
            
            biases = bias('biases', [output_dim])
            #对卷积层添加偏置后保持原有张量尺寸不变
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            p += [kernel, biases]
            activation = relu(conv, name+'_relu')
            return activation
    else:
        #选择加入BN层的时候
        strides = [1, d_h, d_w, 1]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            #见上方注释，此处实际上是使用之前定义的随机权重函数生成output_dim个权重随机的卷积核
            kernel = weight(
                'kernel_w',
                [k_h, k_w, input_op.get_shape()[-1], output_dim]
            ) #input_op.get_shape()[-1]为反向读取input_op的最后一个维度的大小

            conv = tf.nn.conv2d(input_op, kernel, strides=strides, padding='SAME')
            #padding: SAME   用0填充边界，使得每次卷积之后图像尺寸不变，一般需要接一个池化层来对图像降维
            #         VALID  不填充边界，每次卷积后图像尺寸发生改变，改变大小根据卷积核大小而定
            
            biases = bias('biases', [output_dim])
            #对卷积层添加偏置后保持原有张量尺寸不变
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            p += [kernel, biases]
            conv_with_bn = batch_norm(conv, 'bn')
            activation = relu(conv_with_bn, 'bn_relu')
            return activation


#全连接层(Dense)
def fc_layer(input_op, name, output_shape, p):
    #对一个张量(tensor)使用get_shape()来获取张量的大小，get_shape()返回一个元组
    #元组与list类似，但是元组中的元素不能修改
    #as_list()将元组转换为list
    input_shape = input_op.get_shape()[-1].value

    #以下生成的所有variable都处于variabel_scope(name)中
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        kernel = weight(
            name+'_w', shape=[input_shape, output_shape], stddev=0.02,
            initializer_weight=tf.contrib.layers.xavier_initializer_conv2d()
            )
        biases  = bias ('biases', [output_shape], 0.1)

        # tf.nn.relu_layer()用来对输入变量input_op与kernel做乘法并且加上偏置b
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=name)
        p+=[kernel, biases]
        return activation


#池化层
def mpool_layer(input_op, name='max_pool', k_h=2, k_w=2, d_h=2, d_w=2):
    '''
    定义maxpool层
    input：input_op
    池化尺寸：k_h x k_w
    步长：d_h x d_w
    padding：SAME
    padding="SAME" 输出尺寸为W/S(其中W为输入尺寸,S为stride)
    padding="VALID" 输出尺寸为(W-F+1)/S(其中F为卷积核尺寸)
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return tf.nn.max_pool(input_op, ksize=[1,k_h,k_w,1],
                                strides=[1,d_h, d_w, 1],
                                padding='SAME',
                                name=name)

#dropout层
def dropout(input_op, name='dropout', keep_prob=0.7):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return tf.nn.dropout(input_op, keep_prob=keep_prob)


#flatten层，把输入展开成一维向量，从而输入全连接层
def flatten_layer(input_op, name='flatten'):
    shape = input_op.get_shape()
    flatten_shape = shape[1].value * shape[2].value * shape[3].value
    with tf.variable_scope(name, reuse=tf.AUTO_REUSEw):
        return tf.reshape(input_op, [-1, flatten_shape], name=name)


#串联约束条件到feature map
def conv_cond_concat(input_op, name, cond):
    input_op_shape = input_op.get_shape().as_list()
    cond_shape  = input_op.get_shape().as_list()

    #本质上是一个张量拼接的过程
    # 在第4个维度上（feature map 维度上）把条件和输入串联起来，(axis=3)
    # 条件会被预先设为四维张量的形式，假设输入为 [64, 32, 32, 32] 维的张量，
    # 条件为 [64, 32, 32, 10] 维的张量，那么输出就是一个 [64, 32, 32, 42] 维张量
    #t1 = [[1, 2, 3], [4, 5, 6]]
    #t2 = [[7, 8, 9], [10, 11, 12]]
    #按照第0维连接
    #tf.concat( [t1, t2]，0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    #按照第1维连接
    #tf.concat([t1, t2]，1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return tf.concat(
            [input_op, cond*tf.ones(input_op_shapes[0:3]+cond_shape[3:])], 3)


#训练时使用的评估指标

#损失函数
def loss_op(logits, label_batches):
    with tf.variable_scope('loss_op', reuse=tf.AUTO_REUSE):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batches)
        cost = tf.reduce_mean(cross_entropy)
        return cost

#评价分类精确度函数
def accuracy_op(logits, labels):
    '''
    tf.nn.in_top_k(predictions, targets, k, name=None)
        predictions：预测的结果，预测矩阵大小为样本数×标注的label类的个数的二维矩阵。
        targets：实际的标签，大小为样本数。
        k：每个样本的预测结果的前k个最大的数里面是否包含targets预测中的标签，一般都是取1，即取预测最大概率的索引与标签对比。
    '''
    with tf.variable_scope('accuracy_op', reuse=tf.AUTO_REUSE):
        acc = tf.nn.in_top_k(logits, labels, 1)
        acc = tf.cast(acc, tf.float32)
        acc = tf.reduce_mean(acc)
        return acc

#训练时的优化器
def AdamOptimizer(loss, learning_rate=1e-4):
    with tf.variable_scope('Adam', reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return optimizer