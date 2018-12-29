#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2018/11/23
Author: Xu Yucheng
Abstract: Code for load image data and label to .tfrecord files
          and write all train image & valid image path to .txt files
'''
import os
import cv2
import sys
import shutil
import numpy as np
import tensorflow as tf 
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.data import Dataset

from utils import clock, resize_image, Bubble_Sort, Get_ID

DATABASE = '/home/kamerider/Documents/DataBase'
TESTDATA_DIR = '/home/kamerider/Documents/TestData'
TRAIN_DATA_DIR = '/home/kamerider/Documents/Dataset_TF/dataset_train'
VALID_DATA_DIR = '/home/kamerider/Documents/Dataset_TF/dataset_valid'
TFRECORD_DIR = '../TFRecords'


#TESTDATA_DIR = '/home/kamerider/Documents/small_test'


IMAG_SIZE = 64
minimun_id = 9999999
#order_sheet to generate one-hot sheet
order_sheet = np.array ([], dtype=np.int32)
#number of predefined classes
predefined_class = None

#储存每一张图片的绝对路径以及对应的标签
train_image_paths = []
train_labels = []

valid_image_paths = []
valid_labels = []

R_value = []
G_value = []
B_value = []

RGB_MEAN = []


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def Generate_OrderSheet ():
    '''
    初始得到的标签列表元素为string类型
    首先将其转化为np.array
    然后根据得到的np.array中元素的大小，从小到大排序得到一张次序表
    然后遍历np.array，每一个元素在次序表中搜索对应，并用次序表中这个元素的下标
    来取代np.array中该元素的值
    这样就可以把一个string类型的列表转化为一个one-hot编码方式所需求得顺序np.array数组
    '''
    #using os operation to get the number of classes
    print ("============================================================================")
    print ('||                 the original class labels is:                          ||')
    print ("============================================================================")
    #print labels
    folders = os.listdir (DATABASE)
    folders = np.asarray (folders, dtype = np.float32)
    print (folders)
    print ("============================================================================\n")
    global minimun_id
    minimun_id = min(folders)
    print ("=======================================")
    print ('the minimum class label is: ' + str(minimun_id) + '||')
    print ("=======================================\n")
    

    class_num = len(folders)
    global predefined_class
    predefined_class = class_num
    print ("=================================================")
    print ('the total number of predefined classes is: ' + str(class_num) + '||')
    print ("=================================================\n")
    relative_sheet = np.array ([], dtype=np.uint32)
    for folder in folders:
        temp = int(folder) - minimun_id
        relative_sheet = np.append (relative_sheet, temp)
    print ("=============================================================================")
    print ('||             the relative class label sheet is:                          ||')
    print ("=============================================================================")
    print (relative_sheet)
    print ("=============================================================================\n")
    global order_sheet
    order_sheet = Bubble_Sort (relative_sheet)
    print ("=============================================================================")
    print ('||                  generated order sheet:                                 ||')
    print ("=============================================================================")
    print (order_sheet)
    print ("=============================================================================\n")

    return order_sheet

def rgb_mean(image):
    im_r = image[:,:,0]
    im_g = image[:,:,1]
    im_b = image[:,:,2]

    im_r_mean = np.mean(im_r)
    im_g_mean = np.mean(im_g)
    im_b_mean = np.mean(im_b)

    R_value.append(im_r_mean)
    G_value.append(im_g_mean)
    B_value.append(im_b_mean)

@clock
def find_train_image(train_path):
    is_visited = True
    for File in os.listdir(train_path):
        full_path = os.path.abspath(os.path.join(train_path, File))
        if os.path.isdir(full_path):
            is_visited = True
            find_train_image(full_path)
        else:
            if File.endswith('.png') or File.endswith('.PNG'):
                folder_path = os.path.abspath(os.path.join(full_path, "../"))
                upper_folder_path = os.path.abspath(os.path.dirname(folder_path))
                curr_id = int(Get_ID(folder_path, upper_folder_path))
                for i in range(len(order_sheet)):
                    temp = curr_id - minimun_id
                    if temp == order_sheet[i]:
                        onehot_id = i
                if is_visited == True:
                    print ("[INFO] student id: %d ----> one_hot label: %d"%(curr_id, onehot_id))
                    is_visited = False
                train_image_paths.append(full_path)
                train_labels.append(onehot_id)

                #计算每一张图片上RGB三通道的均值
                image = cv2.cvtColor (
                        resize_image(cv2.imread(full_path)), cv2.COLOR_BGR2RGB
                    )
                rgb_mean(image)
                
@clock
def find_valid_image(valid_path):
    is_visited = True
    for File in os.listdir(valid_path):
        full_path = os.path.abspath(os.path.join(valid_path, File))
        if os.path.isdir(full_path):
            is_visited = True
            find_valid_image(full_path)
        else:
            if File.endswith('.png') or File.endswith('.PNG'):
                folder_path = os.path.abspath(os.path.join(full_path, "../"))
                upper_folder_path = os.path.abspath(os.path.dirname(folder_path))
                curr_id = int(Get_ID(folder_path, upper_folder_path))

                for i in range(len(order_sheet)):
                    temp = curr_id - minimun_id
                    if temp == order_sheet[i]:
                        onehot_id = i

                if is_visited == True:
                    print ("[INFO] student id: %d ----> one_hot label: %d"%(curr_id, onehot_id))
                    is_visited = False

                valid_image_paths.append(full_path)
                valid_labels.append(onehot_id)

def write_path_to_txt(image_path, label, name):
    #先判断目标文件夹是否存在，不存在的话创建文件夹
    DES_DIR = os.path.abspath(os.path.join(os.getcwd(), TFRECORD_DIR))
    if not os.path.exists(DES_DIR):
        os.makedirs(DES_DIR)
    if name == 'train':
        filename = os.path.abspath(os.path.join(TFRECORD_DIR, "train.txt"))
        with open(filename, "w") as f:
            print ("[RESULT] Number of train images: %d"%(len(train_labels)))
            for index in range(len(train_labels)):
                f.write(str(train_image_paths[index]) + " " + str(train_labels[index]) + "\n")
            f.close()

    if name == 'valid':
        filename = os.path.abspath(os.path.join(TFRECORD_DIR, "valid.txt"))
        with open(filename, "w") as f:
            print ("[RESULT] Number of valid images: %d"%(len(valid_labels)))
            for index in range(len(valid_labels)):
                f.write(str(valid_image_paths[index]) + " " + str(valid_labels[index]) + "\n")
            f.close()


def extract_image(filename):
    image = cv2.cvtColor (
                        resize_image(cv2.imread(filename)), cv2.COLOR_BGR2RGB
                    )
    return image

def generate_tfrecord_file():
    #先判断目标文件夹是否存在，不存在的话创建文件夹
    DES_DIR = os.path.abspath(os.path.join(os.getcwd(), TFRECORD_DIR))
    if not os.path.exists(DES_DIR):
        os.makedirs(DES_DIR)

    #生成train和valiad两个.tfrecord文件
    train_filename = os.path.abspath(os.path.join(TFRECORD_DIR, "train.tfrecords"))
    with tf.python_io.TFRecordWriter(train_filename) as tfrecord_writer:
        for i in range(len(train_labels)):
            #使用tf.gfile.FastGFile()读取图片文件
            image_raw = tf.gfile.FastGFile(train_image_paths[i], 'rb').read()
            label = train_labels[i]

            #create features dict
            features = {
                'image_raw': _bytes_feature(image_raw),
                'label': _int64_feature(label)
            }
            #将所有的feature和成为一整个features
            tf_features = tf.train.Features(feature=features)

            #create example protocol buffer
            tf_example = tf.train.Example(features=tf_features)

            #序列化样本
            tf_serialized = tf_example.SerializeToString()
            #写入样本
            tfrecord_writer.write(tf_serialized)
    tfrecord_writer.close()

    #记录验证集数据
    valid_filename = os.path.abspath(os.path.join(TFRECORD_DIR, "valid.tfrecords"))
    with tf.python_io.TFRecordWriter(valid_filename) as tfrecord_writer:
        for i in range(len(valid_labels)):
            #使用tf.gfile.FastGFile()读取图片文件
            image_raw = tf.gfile.FastGFile(valid_image_paths[i], 'rb').read()
            label = valid_labels[i]

            #create features dict
            features = {
                'image_raw': _bytes_feature(image_raw),
                'label': _int64_feature(label)
            }
            #将所有的feature和成为一整个features
            tf_features = tf.train.Features(feature=features)

            #create example protocol buffer
            tf_example = tf.train.Example(features=tf_features)

            #序列化样本
            tf_serialized = tf_example.SerializeToString()
            #写入样本
            tfrecord_writer.write(tf_serialized)
    tfrecord_writer.close()

def load_data_main():
    Generate_OrderSheet()
    print (order_sheet)
    find_train_image(TRAIN_DATA_DIR)
    find_valid_image(VALID_DATA_DIR)

    temp_means = [R_value, G_value, B_value]
    for mean in temp_means:
        RGB_MEAN.append(np.mean(mean))
    print ("[RESULT] RGB mean of all train_data is: [ %f, %f, %f ]"%(RGB_MEAN[0], RGB_MEAN[1], RGB_MEAN[2]))


    write_path_to_txt(train_image_paths, train_labels, "train")
    write_path_to_txt(valid_image_paths, valid_labels, "valid")

    generate_tfrecord_file()

if __name__ == '__main__':
    main()
    
