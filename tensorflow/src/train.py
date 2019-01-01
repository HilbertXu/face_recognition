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
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

#user defined modules
from load_data import load_data_main, RGB_MEAN
from model import VGG
from ops import loss_op, accuracy_op, AdamOptimizer
from utils import *
from image_preprocess import *
from tqdm import tqdm

TFRECORD_DIR = '/home/kamerider/machine_learning/face_recognition/tensorflow/TFRecords'
filewriter_path = "/home/kamerider/machine_learning/face_recognition/tensorflow/tensorboard"  # 存储tensorboard文件
checkpoint_path = "/home/kamerider/machine_learning/face_recognition/tensorflow/history"  # 训练好的模型和参数存放目录

train_data_size=0
valid_data_size=0

train_batch_num=0
valid_batch_num=0
class_num=0

BATCH_SIZE=64
NUM_EPOCHES=200
IMAGE_SIZE=64

lrate = [0.001, 0.0005, 0.0001]
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
	#train_files = tf.train.match_filenames_once(train_file_path)
	#valid_files = tf.train.match_filenames_once(valid_file_path)

	#定义train dataset和对应的迭代器
	dataset_train = tf.data.TFRecordDataset(train_file_path)
	#shuffle buffer 大小设置为所有训练数据的数量，使整个训练集全部打乱
	dataset_train = dataset_train.shuffle(train_data_size)
	#设置整个dataset_train进入网络的次数，即epoches
	#由于在下面使用了for循环来控制epoch，所以不再需要repeat操作
	#dataset_train = dataset_train.repeat(NUM_EPOCHES)
	#使用双线程调用解析函数来解析储存在dataset_train中的proto_example

	dataset_train = dataset_train.map(_parse_tfrecord)
	#dataset_train = dataset_train.map(lambda image, label:(image_preprocess(image, IMAGE_SIZE, IMAGE_SIZE), label), num_parallel_calls=2)
	dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
	dataset_train = dataset_train.prefetch(5)
	#train_iterator = dataset_train.make_initializable_iterator()
	print ("Batch_Size is: %d"%(BATCH_SIZE))

	# 创建 validation dataset 和对应的迭代器
	dataset_valid = tf.data.TFRecordDataset(valid_file_path)
	dataset_valid = dataset_valid.map(_parse_tfrecord)
	#dataset_valid = dataset_valid.map(lambda image, label: (image_preprocess(image, IMAGE_SIZE, IMAGE_SIZE), label), num_parallel_calls=2)
	dataset_valid = dataset_valid.batch(BATCH_SIZE, drop_remainder=True)
	#valid_iterator = dataset_valid.make_initializable_iterator()

	return dataset_train, dataset_valid

def run_vgg_training():
	#训练阶段
	train_data_size, valid_data_size = check_dataset()
	#获取训练集和验证集的迭代器，这个迭代器会在每一个epoch更新
	train_data, valid_data = read_train_valid_tfrecord()

	train_iterator = train_data.make_initializable_iterator()
	valid_iterator = valid_data.make_initializable_iterator()

	train_batch = train_iterator.get_next()
	valid_batch = valid_iterator.get_next()

	#如果使用initializable_iterator()的时候需要初始化迭代器
	#sess.run(train_iterator.initializer)
	#sess.run(valid_iterator.initializer)
	

	#网络训练部分
	#使用占位符表示每个batch输入网络的图片和标签
	x_train = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3])
	y_train = tf.placeholder(tf.float32, [BATCH_SIZE, class_num])
	learning_rate = tf.placeholder(tf.float32)
	predicts, logits, _ = VGG(x_train, class_num=class_num)
	cost = loss_op(logits, y_train)
	optimizer = AdamOptimizer(cost,learning_rate=learning_rate)
	accuracy = accuracy_op(predicts, y_train)

	#网络验证部分
	x_valid = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3])
	y_valid = tf.placeholder(tf.float32, [BATCH_SIZE, class_num])
	val_predicts, val_logits, _ = VGG(x_valid, class_num=class_num)
	val_cost = loss_op(val_logits, y_valid)
	val_accuracy = accuracy_op(val_predicts, y_valid)

	#tensorboard 可视化
	tf.summary.scalar('cross_entropy', cost)
	tf.summary.scalar('accuracy', accuracy)
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter(filewriter_path)
	saver = tf.train.Saver()
	
	#训练函数
	with tf.Session() as sess:
		
		print("[INFO] Open tensorboard at --logdir %s"%(filewriter_path))
		for epoch in range(NUM_EPOCHES):
			#把模型图加载进tensorflow
			writer.add_graph(sess.graph)

			#每一个epoch初始化一次训练集和验证集的迭代器
			sess.run(train_iterator.initializer)
			sess.run(valid_iterator.initializer)
			sess.run(tf.global_variables_initializer())
				
			#储存每一个epoch中的各项数据	
			epoch_loss=0
			epoch_acc=0
			epoch_val_loss=0
			epoch_val_acc=0
			train_batch_count = 1
			valid_batch_count = 1
			print("[TRAINING...（¯﹃¯）] Start training {}/{} epoch at time: {}".format(epoch+1, NUM_EPOCHES, datetime.now()))

			#学习率调整
			if 0 <= epoch <= change_epoch[0]:
				rate = lrate[0]
			elif change_epoch[0] <= epoch <= change_epoch[1]:
				rate = lrate[1] 
			else:
				rate = lrate[2]
			
			#使用feed_dict将生成的每一个batch数据输入网络进行训练
			while True:		
				try:
					train_image_batch, train_label_batch = sess.run(train_batch)
					#注意左侧变量不能与右侧图运算的变量重名，否则会改变图变量的类型，使得计算不能继续
					loss, _, acc = sess.run(
						[cost, optimizer, accuracy],
						feed_dict={
							x_train: train_image_batch,
							y_train: train_label_batch,
							learning_rate: rate
						})
						
					epoch_loss += loss
					epoch_acc += acc
					train_batch_count+=1
					#print ("batch_num %d"%(batch_num))
					if train_batch_count % 100==0:
						#每100个batch输出一次平均loss和acc
						print ("[Batch number %d/%d]"%(train_batch_count, train_batch_num))
						print ("[TRAINING... (￣^￣)] Training Loss: %f \t Training Accuracy: %f"%(
						loss, acc
					))
				except tf.errors.OutOfRangeError:
					epoch_loss=epoch_loss/train_batch_count
					epoch_acc=epoch_acc/train_batch_count
					print ("[TRAINING... (￣^￣)] Training Loss in epoch %d/%d is: %f \t Training Accuracy in epoch %d/%d is: %f"%(
						epoch+1, NUM_EPOCHES, epoch_loss, epoch+1, NUM_EPOCHES, epoch_acc
					))
					break

			#每一个batch的训练结束后，生成验证数据进行验证
			while True:
				try:
					valid_image_batch, valid_label_batch = sess.run(valid_batch)
					val_loss, val_acc = sess.run(
						[val_cost, val_accuracy],
						feed_dict={
							x_valid: valid_image_batch,
							y_valid: valid_label_batch
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
	run_vgg_training()


    
