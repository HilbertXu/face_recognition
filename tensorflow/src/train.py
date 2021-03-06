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
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

#user defined modules
from load_data import load_data_main, RGB_MEAN
from model import VGG
from ops import loss_op, accuracy_op, AdamOptimizer, sgdOptimizer, momentumOptimizer
from utils import *
from vgg_preprocessing import *
from tqdm import tqdm

TFRECORD_DIR = '../TFRecords'
filewriter_path = "../tensorboard"  # 存储tensorboard文件
checkpoint_path = "../history"  # 训练好的模型和参数存放目录

train_data_size=0
valid_data_size=0

train_batch_num=0
valid_batch_num=0
class_num=0

BATCH_SIZE=64
NUM_EPOCHES=100
IMAGE_SIZE=64

lrate = [0.0001, 0.00005, 0.000001]
change_epoch = [20, 50]

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
			global train_batch_num
			train_batch_num = int(train_data_size/BATCH_SIZE)
			global class_num
			class_num = int(f.readline())
		f.close()

		with open(os.path.abspath(os.path.join(TFRECORD_DIR, "valid.txt")), 'r') as f:
			global valid_data_size
			valid_data_size = int(f.readline())
			global valid_batch_num
			valid_batch_num = int(valid_data_size/BATCH_SIZE)
		f.close()
	return train_data_size, valid_data_size



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
	#数据增强
	image_decoded = tf.image.random_flip_left_right(image_decoded)
	#image_decoded = tf.image.per_image_standardization(image_decoded)
	#图像归一化
	image_decoded = tf.cast(image_decoded, tf.float32)/255.0

	#处理图像对应的标签，将其转化为one-hot code
	label = tf.cast(label, tf.int32)
	label = tf.one_hot(label, depth=class_num, on_value=1)
	return image_decoded, label

def read_train_valid_tfrecord():
	train_file_path = os.path.abspath(os.path.join(TFRECORD_DIR, "train.tfrecords"))
	valid_file_path = os.path.abspath(os.path.join(TFRECORD_DIR, "valid.tfrecords"))
	
	print ("[INFO] Reading dataset from .tfrecord files")
	#定义train dataset和对应的迭代器
	dataset_train = tf.data.TFRecordDataset(train_file_path)
	#shuffle buffer 大小设置为所有训练数据的数量，使整个训练集全部打乱
	dataset_train = dataset_train.shuffle(train_data_size)
	dataset_train = dataset_train.map(_parse_tfrecord)
	dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
	dataset_train = dataset_train.prefetch(5)
	print ("Batch_Size is: %d"%(BATCH_SIZE))

	# 创建 validation dataset 和对应的迭代器
	dataset_valid = tf.data.TFRecordDataset(valid_file_path)
	dataset_valid = dataset_valid.map(_parse_tfrecord)
	dataset_valid = dataset_valid.batch(BATCH_SIZE, drop_remainder=True)

	return dataset_train, dataset_valid

def run_vgg_training():
	#训练阶段
	train_data_size, valid_data_size = check_dataset()
	#网络训练部分
	#使用占位符表示每个batch输入网络的图片和标签
	x_train = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3])
	y_train = tf.placeholder(tf.float32, [BATCH_SIZE, class_num])
	#learning_rate = tf.placeholder(tf.float32)

	predicts, softmax_output, logits, fc_params_loss = VGG(x_train, class_num=class_num)
	cost = loss_op(softmax_output, y_train, fc_params_loss)
	cost += fc_params_loss

	#列出所有可以训练的参数
	var_list = [v for v in tf.trainable_variables()]
	#train_op = sgdOptimizer(tf.nn.l2_loss(cost), params, learning_rate=learning_rate)
	#train_op = sgdOptimizer(cost, learning_rate=learning_rate)
	#train_op = sgdOptimizer(cost, var_list, learning_rate=learning_rate)
	#train_op = AdamOptimizer(cost, learning_rate=learning_rate)
	train_op = momentumOptimizer(cost)
	accuracy = accuracy_op(predicts, y_train)

	#tensorboard 可视化
	tf.summary.scalar('cross_entropy', cost)
	tf.summary.scalar('accuracy', accuracy)
	merged_summary = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(filewriter_path + '/train')
	valid_writer = tf.summary.FileWriter(filewriter_path + '/valid')
	saver = tf.train.Saver()

	'''
	@TODO
	修改数据输入模式
	设置为只搭建一次网络，通过控制feed_dict输入内容来控制输出的是训练结果还是验证结果
	这样就不需要上方的val_*变量，使代码得到精简
	'''
	
	#训练函数
	with tf.Session() as sess:
		
		print("[INFO] Open tensorboard at --logdir %s"%(filewriter_path))
		for epoch in range(NUM_EPOCHES):
			#获取训练集和验证集的迭代器，这个迭代器会在每一个epoch更新
			train_data, valid_data = read_train_valid_tfrecord()

			#如果使用initializable_iterator()的时候需要初始化迭代器
			train_iterator = train_data.make_initializable_iterator()
			valid_iterator = valid_data.make_initializable_iterator()

			train_batch = train_iterator.get_next()
			valid_batch = valid_iterator.get_next()
			#把模型图加载进tensorflow
			train_writer.add_graph(sess.graph)

			#每一个epoch初始化一次训练集和验证集的迭代器
			sess.run(train_iterator.initializer)
			sess.run(valid_iterator.initializer)
			sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
				
			#储存每一个epoch中的各项数据	
			epoch_loss=0
			epoch_acc=0
			epoch_val_loss=0
			epoch_val_acc=0
			train_batch_count = 1
			valid_batch_count = 1
			print("[TRAINING...（¯﹃¯）] Start training {}/{} epoch at time: {}".format(epoch+1, NUM_EPOCHES, datetime.now()))

			#使用feed_dict将生成的每一个batch数据输入网络进行训练
			while True:		
				try:
					train_image_batch, train_label_batch = sess.run(train_batch)
					sess.run(
						train_op,
						feed_dict={
								x_train: train_image_batch,
								y_train: train_label_batch
							})
					if train_batch_count % 10==0:
						#每100个batch输出一次平均loss和acc
						loss, acc = sess.run(
							[cost, accuracy],  
							feed_dict={
								x_train: train_image_batch,
								y_train: train_label_batch
						})
						print ("[EPOCH %d/%d  Batch number %d/%d]"%(epoch+1, NUM_EPOCHES, train_batch_count, train_batch_num))
						print ("[TRAINING... (￣^￣)] Training Loss: %f \t Training Accuracy: %f"%(
						loss, acc
					))
					if train_batch_count % 50==0:
						

						run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
						run_metadata = tf.RunMetadata()
						summary, _ = sess.run(
							[merged_summary, train_op], 
							feed_dict={
								x_train: train_image_batch,
								y_train: train_label_batch
							},
							options=run_options,
							run_metadata=run_metadata)
						train_writer.add_run_metadata(run_metadata, 'Epoch %03d Batch %03d' %(train_batch_count, epoch))
						train_writer.add_summary(summary, train_batch_count)
						print('Adding run metadata for', train_batch_count)  
					train_batch_count+=1
				except tf.errors.OutOfRangeError:
					saver.save(sess, checkpoint_path+'/model.ckpt', epoch+1)
					break
			train_writer.close()

			#每一个batch的训练结束后，生成验证数据进行验证
			while True:
				try:
					valid_image_batch, valid_label_batch = sess.run(valid_batch)

					#改变feed_dict来获得验证数据
					val_loss, val_acc = sess.run(
						[cost, accuracy],
						feed_dict={
							x_train: valid_image_batch,
							y_train: valid_label_batch
						})
					epoch_val_loss+=val_loss
					epoch_val_acc+=val_acc	
					valid_batch_count+=1
				except tf.errors.OutOfRangeError:
					epoch_val_loss = epoch_val_loss/valid_batch_count
					epoch_val_acc = epoch_val_acc/valid_batch_count
					print ("[VALIDATION Σ(っ°Д°;)っ] Validation Loss in epoch %d/%d is: %f \t Validation Accuracy in epoch %d/%d is: %f"%(
						epoch+1, NUM_EPOCHES, epoch_val_loss, epoch+1, NUM_EPOCHES, epoch_val_acc
					))
					print ("[EPOCH (ง •̀_•́)ง] epoch %d/%d has finished!"%(epoch+1, NUM_EPOCHES))
					break

if __name__ == '__main__':
	os.system("rm -rf /home/kamerider/machine_learning/face_recognition/tensorflow/tensorboard/train/*")
	os.system("rm -rf /home/kamerider/machine_learning/face_recognition/tensorflow/tensorboard/valid/*")
	os.system("rm -rf /home/kamerider/machine_learning/face_recognition/tensorflow/history/*")
	run_vgg_training()


    
