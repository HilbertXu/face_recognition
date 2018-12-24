# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
Date: 2018/12/25
Author: Xu Yucheng
Abstract: Code for rearrange dataset and split data 7/3 manually
'''

import os
import cv2
import sys
import shutil
import random

#要求的储存格式但未分割的数据集根目录（这一个中转的文件夹）
#DES_DIR/16114xxx/images
DES_DIR = '/home/kamerider/temp'

#最终要求的数据集储存方式
#TRAIN_DIR/1611xxx/images(700 pics)
#VALID_DIR/1611xxx/images(300 pics)
TRAIN_DIR = '/home/kamerider/to/train_data'
VALID_DIR = '/home/kamerider/to/valid_data'

#储存格式不正确的数据集的根目录
#FROM_DIR/1611xxx/01/images
FROM_DIR = '/home/kamerider/source'
DATASET = '/home/kamerider/dataset'

labels=[]

def Get_ID (str1, str2):
    '''
    通过路径操作(从str1中删去str2部分)
    得到每张照片对应的学号
    '''
    length = len(str2)
    label = str1[length+1:]
    return label


def Find_Image (path):
    is_visited = True
    if not os.path.exists(DES_DIR):
        os.chdir(os.path.abspath(os.path.join(DES_DIR,"../")))
        os.mkdir("to")
    for File in os.listdir (path):
        full_path = os.path.abspath (os.path.join(path,File))
        if os.path.isdir (full_path):
            is_visited = True
            Find_Image (full_path)
        else:
            if File.endswith('.png') or File.endswith('.PNG'):
                if cv2.imread(full_path).shape[0] < 500:
                    #print 'find face image'+full_path
                    face_body_path = os.path.abspath(os.path.join((full_path),"../"))
                    folder_path = os.path.abspath(os.path.join((full_path),"../.."))
                    upper_folder_path = os.path.abspath(os.path.dirname(folder_path))
                    #get filename & fileID
                    curr_id = Get_ID(folder_path, upper_folder_path)
                    DES_PATH = DES_DIR+'/'+curr_id
                    if not os.path.exists(DES_PATH):
                        os.chdir(DES_DIR)
                        os.mkdir(curr_id)
                    else:
                        shutil.copy(full_path, DES_PATH)
                    
                    if is_visited:
                        labels.append(curr_id)
                        print ("\033[0;31;40m Images in %s have been moved to %s\033[0m"%(face_body_path, DES_PATH))
                        is_visited = False
    return 0

def read_image(path):
    #先将图片全部读取到image里面，然后打乱image
    #现在图片储存在/home/kamerider/to文件夹中
    #储存路径为：/to/1611xxx/images
    for label in labels:
        #遍历每一个文件夹
        print "current label is " + label
        #更新训练集和测试集的绝对路径
        train_path = os.path.abspath(os.path.join(DATASET, 'dataset_train'))
        valid_path = os.path.abspath(os.path.join(DATASET, 'dataset_test'))
        current_path = os.getcwd()
        if not os.path.exists(train_path) or not os.path.exists(valid_path):
            print("Create /dataset_train and /dataset_test in %s"%(DATASET))
            os.chdir(DATASET)
            os.mkdir('dataset_train')
            os.mkdir('dataset_test')
            os.chdir(current_path)
        #在dataset_train & dataset_test下建立以每个学号命名的文件夹
        if not os.path.exists(os.path.abspath(os.path.join(train_path,label))) or not os.path.exists(os.path.abspath(os.path.join(valid_path,label))):
            print ("Create /%s in /dataset_train and /dataset_test"%(label))
            os.chdir(train_path)
            os.mkdir(label)
            os.chdir(valid_path)
            os.mkdir(label)
            os.chdir(current_path)
        
        #更新训练集和测试集的绝对路径
        train_path = os.path.abspath(os.path.join(train_path,label))
        valid_path = os.path.abspath(os.path.join(valid_path,label))

        folder_path = os.path.abspath(os.path.join(DES_DIR, label))
        
        print ("Reading images from %s"%(folder_path))
        
        count = 0
        for File in os.listdir(folder_path):
            #遍历每一个文件夹内的照片并储存
            full_path = os.path.abspath(os.path.join(folder_path, File))
            filename = Get_ID(full_path,os.path.abspath(os.path.join(full_path,"../")))
            train_image_path = os.path.abspath(os.path.join(train_path, filename))
            valid_image_path = os.path.abspath(os.path.join(valid_path, filename))
            count +=1

            if count <= 700:
                shutil.move(full_path, train_image_path)
            else:
                shutil.move(full_path, valid_image_path)


        
        #读完一个文件夹的图片之后将图片打乱并7/3分割
        #然后储存到指定路径下的train_data和valid_data文件夹中

                
    shutil.rmtree(DES_DIR)
    return 0
                    
if __name__ == '__main__':
    Find_Image(FROM_DIR)
    print labels
    read_image(DATASET)
    
                    
