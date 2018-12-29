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

def resize_image (image):
        return cv2.resize (image, (64, 64))
    
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