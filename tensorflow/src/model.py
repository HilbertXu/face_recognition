# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
Date: 2018/12/29
Author: Xu Yucheng
Abstract: Code for build VGG-16 model
'''
import time
import tensorflow as tf
import math
from datetime import datetime
from ops import *

EGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数

#input_op: 每一个batch输入网络的图像数据
#keep_prob: drop_out 层中保留神经元链接的比例，在测试的时候赋值为1 
def VGG(input_op, keep_prob=0.5, class_num=62):
	#通过依次采用多个3x3的卷积核，模仿出更大的感受野的效果
	p=[]
	#第一层卷积层用作输入层
	#此后经过卷积核逐渐增多的5次卷积
	tf.summary.image('images', input_op, 64)
	conv1_1, conv1_1_w = conv2d(input_op, name='conv1_1', output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
	conv1_2, conv1_2_w = conv2d(conv1_1, name='conv1_2', output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
	mpool1 = mpool_layer(conv1_2, name='mpool1', k_h=2, k_w=2, d_h=2, d_w=2)
	conv1_drop = dropout(mpool1, name='conv1_drop', keep_prob=keep_prob)

	conv2_1, conv2_1_w = conv2d(conv1_drop, name='conv2_1', output_dim=128, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
	conv2_2, conv2_2_w = conv2d(conv2_1, name='conv2_2', output_dim=128, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
	mpool2 = mpool_layer(conv2_2, name='mpool2', k_h=2, k_w=2, d_h=2, d_w=2)
	conv2_drop = dropout(mpool2, name='conv2_drop', keep_prob=keep_prob)

	conv3_1, conv3_1_w = conv2d(conv2_drop, name='conv3_1', output_dim=256, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
	conv3_2, conv3_2_w = conv2d(conv3_1, name='conv3_2', output_dim=256, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
	conv3_3, conv3_3_w = conv2d(conv3_2, name='conv3_3', output_dim=256, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
	mpool3 = mpool_layer(conv3_3, name='mpool3', k_h=2, k_w=2, d_h=2, d_w=2)
	conv3_drop = dropout(mpool3, name='conv3_drop', keep_prob=keep_prob)

	conv4_1, conv4_1_w = conv2d(conv3_drop, name='conv4_1', output_dim=512, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
	conv4_2, conv4_2_w = conv2d(conv4_1, name='conv4_2', output_dim=512, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
	conv4_3, conv4_3_w = conv2d(conv4_2, name='conv4_3', output_dim=512, k_h=3, k_w=3, d_h=1, d_w=1, p=p)
	mpool4 = mpool_layer(conv4_3, name='mpool4', k_h=2, k_w=2, d_h=2, d_w=2)
	conv4_drop = dropout(mpool4, name='conv4_drop', keep_prob=keep_prob)

	flatten = flatten_layer(conv4_drop, name='flatten')

	fc6, fc6_w, fc6_b = fc_layer(flatten, name='fc6', output_shape=4096, p=p)
	fc6_drop = dropout(fc6, name='fc6_drop', keep_prob=keep_prob)

	fc7, fc7_w, fc7_b = fc_layer(fc6_drop, name='fc7', output_shape=4096, p=p)
	fc7_drop = dropout(fc7, name='fc7_drop', keep_prob=keep_prob)

	#最后一层全连接层需要将之前层的输出激活62个类别对应的62个神经元
	#故fc8的output_shape=62
	fc8, fc8_w, fc8_b = fc_layer(fc7_drop, name='fc8', output_shape=class_num, p=p)

	softmax = softmax_layer(fc8, name='softmax')

	predicitions = tf.argmax(softmax, 1)
	params_loss = l2_regularizer(fc6_w, fc7_w, fc8_w, conv1_1_w, conv1_2_w, 
									conv2_1_w, conv2_2_w, 
									conv3_1_w, conv3_2_w, conv3_3_w,
									conv4_1_w, conv4_2_w, conv4_3_w)
	return predicitions, softmax, fc8, params_loss

def gradients_visualization():
	return 0

def l2_regularizer(fc6_w, fc7_w, fc8_w, conv1_1_w, conv1_2_w, 
									conv2_1_w, conv2_2_w, 
									conv3_1_w, conv3_2_w, conv3_3_w,
									conv4_1_w, conv4_2_w, conv4_3_w):
									
	regularizers = tf.nn.l2_loss(fc6_w) + tf.nn.l2_loss(fc7_w) + tf.nn.l2_loss(fc8_w) +\
					tf.nn.l2_loss(conv1_1_w) + tf.nn.l2_loss(conv1_2_w) +\
					tf.nn.l2_loss(conv2_1_w) + tf.nn.l2_loss(conv2_2_w) +\
					tf.nn.l2_loss(conv3_1_w) + tf.nn.l2_loss(conv3_2_w) + tf.nn.l2_loss(conv3_3_w) +\
					tf.nn.l2_loss(conv4_1_w) + tf.nn.l2_loss(conv4_2_w)	+ tf.nn.l2_loss(conv4_3_w)
	return 1e-4*regularizers


def save_model(sess, save_path):
	with tf.name_scope("saver"):
		saver =  tf.train.Saver()
		saver.save(sess, save_model)


#以下是模拟训练过程并测试模型的部分
def time_tensorflow_run(session,target,feed,info_string):
	num_batches = 100
	num_steps_burn_in = 10  # 预热轮数
	total_duration = 0.0  # 总时间
	total_duration_squared = 0.0  # 总时间的平方和用以计算方差
	for i in range(num_batches + num_steps_burn_in):
		start_time = time.time()
		_ = session.run(target,feed_dict=feed)
		duration = time.time() - start_time
		if i >= num_steps_burn_in:  # 只考虑预热轮数之后的时间
			if not i % 10:
				print('%s:step %d,duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
				total_duration += duration
				total_duration_squared += duration * duration
	mn = total_duration / num_batches  # 平均每个batch的时间
	vr = total_duration_squared / num_batches - mn * mn  # 方差
	sd = math.sqrt(vr)  # 标准差
	print('%s: %s across %d steps, %.3f +/- %.3f sec/batch' % (datetime.now(), info_string, num_batches, mn, sd))

def run_benchmark():
	with tf.Graph().as_default():
		image_size = 64  # 输入图像尺寸
		batch_size = 64
		images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
		keep_prob = tf.placeholder(tf.float32)
		prediction,softmax,fc8,loss, p = VGG(images,keep_prob)
		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		time_tensorflow_run(sess, prediction,{keep_prob:1.0}, "Forward")
		# 用以模拟训练的过程
		objective = tf.nn.l2_loss(fc8)  # 给一个loss
		grad = tf.gradients(objective, p)  # 相对于loss的 所有模型参数的梯度
		time_tensorflow_run(sess, grad, {keep_prob:0.5},"Forward-backward")

if __name__ == '__main__':
	run_benchmark()