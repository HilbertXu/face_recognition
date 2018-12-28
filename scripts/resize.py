#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2018/11/21
Author: Xu Yucheng
Abstract: Code for resize images in folder
'''
import os
import cv2

image_dir = '/home/kamerider/catkin_ws/src/machine_vision/DataBase/1611453/body/'

if __name__ == '__main__':
    index = 10002
    index_max = 10606
    if index < index_max:
        print 'the current image is '+ str(index) +'.png'
        full_path = image_dir + str(index) + '.png'
        frame = cv2.imread(full_path)
        size = frame.shape
        height = size[0]
        width = size[1]
        print 'current image height is ' + str(height)
        print 'current image width is ' + str(width)
        if height != 1920 or width != 1080:
            print 'the current image is '+ str(index) +'.png'
            Size = (1080, 1920)
            res = cv2.resize(frame,Size,interpolation=cv2.INTER_AREA)
            cv2.imwrite(full_path, res)
        index = index + 1
