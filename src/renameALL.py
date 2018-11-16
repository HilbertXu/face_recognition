#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os

#change DIR_PATH to renameALL
BODY_IMAGE_DIR = '/home/kamerider/catkin_ws/src/machine_vision/DataBase/1611472/body/'
FACE_IMAGE_DIR = '/home/kamerider/catkin_ws/src/machine_vision/DataBase/1611472/face/'

def reName(dir_name):
    '''
    实现将所有类别中的所有文章重新命名
    '''
    index = 9999
    tmp = 0
    file_list = os.listdir(dir_name)
    for files in file_list:
        tmp = tmp + 1
        old_dir = os.path.join(dir_name, files)
        print '---------------------------------'
        print 'old path is'+old_dir
        file_name = os.path.splitext(files)[0]
        file_type = os.path.splitext(files)[1]
        temp_dir = os.path.join(dir_name, str(tmp)+file_type)
        print 'using temp file name ' + temp_dir
        os.rename(old_dir, temp_dir)

    temp_list = os.listdir(dir_name)
    print temp_list
    for files in temp_list:
        index = index + 1
        temp_dir = os.path.join(dir_name, files)
        print '---------------------------------'
        print 'temp path is '+temp_dir
        file_name = os.path.splitext(files)[0]
        file_type = os.path.splitext(files)[1]
        new_dir = os.path.join(dir_name, str(index)+file_type)
        print 'using new file name '+new_dir
        os.rename(temp_dir, new_dir)

if __name__ == '__main__':
    print 'Dealing with body images'
    reName(BODY_IMAGE_DIR)
    print 'Dealing with face images'
    reName(FACE_IMAGE_DIR)
