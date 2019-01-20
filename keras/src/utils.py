# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
Date: 2018/11/21
Author: Xu Yucheng 1611453
Abstract: some functions
'''
import os
import sys
import numpy as np
import cv2
import timeit
import matplotlib.pyplot as plt


HISTORY_PATH = '/home/kamerider/machine_learning/face_recognition/keras/History/Train_History.txt'
FIGURE_PATH = '/home/kamerider/machine_learning/face_recognition/'
IMAGE_SIZE = 64

#一个写成装饰器形式的计时器，在函数前加@clock便可以输出每次调用该函数运行时间
def clock(func):
    def clocked(*args):
        t0 = timeit.default_timer()
        result = func(*args)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result
    return clocked

def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE, padding=False):
	#若输入图片是一个非正方形的图片需要扩展边界将其变成一个正方形图片之后再缩放
	if padding:
		top, bottom, left, right = (0, 0, 0, 0)

		#get the size of image
		h, w, _ = image.shape

		#if width != height
		#get the longer one
		longest_edge = max(h, w)

		#calculate how much pixel should be add to the shorter side
		if h < longest_edge:
			dh = longest_edge - h
			top = dh // 2
			bottom = dh - top
		elif w < longest_edge:
			dw = longest_edge - w
			left = dw // 2
			right = dw - left
		else:
			pass
		#set the border color
		BLACK = [0, 0, 0]
		#border
		constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
		#resize image & return
		return cv2.resize(constant, (height, width))
	
	if not padding:
		return cv2.resize(image, (height, width))
    
def Bubble_Sort (array):
        '''
        该函数接受一个储存了所有类别对应学号的数组
        然后对数组进行排序，得到一个从小到大排序的有序数组
        '''
        order_array = array.copy ()
        order_array = sorted(array)
        return order_array
    
def Get_ID (str1, str2):
        '''
        通过路径操作(从str1中删去str2部分)
        得到每张照片对应的学号
        '''
        length = len(str2)
        label = str1[length+1:]
        return label

def Find_minimum_Label (folders):
    '''
    从所有类别中找出id最小的一类
    '''
    minID = 9999999
    for folder in folders:
        int_label = int(folder)
        if int_label < temp_min:
            temp_min = int_label
    return minID

def visualization(hist, nb_epoch):
	print (hist.history.keys())
	with open(HISTORY_PATH,'w') as f:
		f.write(str(hist.history))

	#visualization
	#store the output history
	model_val_loss = hist.history['val_loss']
	model_val_acc  = hist.history['val_acc']
	model_loss     = hist.history['loss']
	model_acc      = hist.history['acc']

	#Using matplotlib to visualize
	epochs = np.arange(nb_epoch)+1
	plt.figure()
	plt.plot(epochs, model_val_loss, label = 'model_val_loss')
	plt.plot(epochs, model_loss, label = 'model_loss')
	plt.title('visualize the training process')
	plt.xlabel('Epoch #')
	plt.ylabel('Validation Loss & Train Loss')
	plt.legend()
	plt.savefig(FIGURE_PATH+'/loss_figure.png')
	plt.show()

	plt.figure()
	plt.plot(epochs, model_val_acc, label = 'model_val_acc')
	plt.plot(epochs, model_acc, label = 'model_acc')
	plt.title('visualize the training process')
	plt.xlabel('Epoch #')
	plt.ylabel('Validation accuracy & Train accuracy')
	plt.legend()
	plt.savefig(FIGURE_PATH+'/acc_figure.png')
	plt.show()

def print_matrix(list):
    for row in range(8):
        if row == 7:
            for col in range(6):
                print (list[row*8+col], end='')
                print ('\t', end='')
            print ('\n')
        else:
            for col in range(8):
                print (list[row*8+col], end='')
                print ('\t', end='')
            print ('\n')