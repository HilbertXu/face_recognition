#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2018/11/23
Author: Xu Yucheng
Abstract: Code for rename images in all folders
'''

import os

#BODY_IMAGE_DIR = '/home/kamerider/catkin_ws/src/machine_vision/DataBase/1611472/body/'
DATA_BASE = '/home/kamerider/Documents/DataBase'

def reName(dir_name):
    '''
    实现将所有类别中的所有文章重新命名
    '''
    index = 9999
    tmp = 0
    count=0
    file_list = os.listdir(dir_name)
    for files in file_list:
        tmp = tmp + 1
        old_dir = os.path.join(dir_name, files)
        #print ('---------------------------------')
        #print ('old path is'+old_dir)
        file_name = os.path.splitext(files)[0]
        file_type = os.path.splitext(files)[1]
        temp_dir = os.path.join(dir_name, str(tmp)+file_type)
        #print ('using temp file name ' + temp_dir)
        os.rename(old_dir, temp_dir)

    temp_list = os.listdir(dir_name)
    for files in temp_list:
        index = index + 1
        count +=1
        temp_dir = os.path.join(dir_name, files)
        #print ('---------------------------------')
        #print ('temp path is '+temp_dir)
        file_name = os.path.splitext(files)[0]
        file_type = os.path.splitext(files)[1]
        new_dir = os.path.join(dir_name, str(index)+file_type)
        if new_dir.endswith('.PNG'):
            #若发现PNG结尾的图片文件，纠正为png
            length = len(new_dir)-4
            temp = new_dir[0:length]
            new_dir =temp + '.png'
        #print ('using new file name '+new_dir)
        os.rename(temp_dir, new_dir)
    return count

if __name__ == '__main__':
    #if need to deal with body image uncomment the following two lines
    #print 'Dealing with body images'
    #reName(BODY_IMAGE_DIR)
    print ('Dealing with face images')
    labels = os.listdir(DATA_BASE)
    for label in labels:
        IMAGE_FOLDER = os.path.abspath(os.path.join(DATA_BASE, label))
        count = reName(IMAGE_FOLDER)
        print ("current folder is %s , %d images in total"%(IMAGE_FOLDER, count))
        
