# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import sys
import numpy as np
import cv2
from keras.utils import np_utils

IMAGE_SIZE = 224
DataBase_Root_Path = '/home/kamerider/machine_learning/face_recognition/DataBase'




#container of images & labels & classes
images = []
labels = []

def resize_image(image):
    return cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE))

def BubbleSort(array):
    order = array.copy()
    order = sorted(array)

    return order

def getLastPath(str1, str2):
    length = len(str2)
    label = str1[length + 1:]
    return label

def findMininumLabel(labels):
    min_label = 9999999
    for label in labels:
        int_label = int(label)
        if int_label < min_label:
            min_label = int_label
    return min_label

def labelStandardlization(labels):
    '''
    初始得到的标签列表元素为string类型
    首先将其转化为np.array
    然后根据得到的np.array中元素的大小，从小到大排序得到一张次序表
    然后遍历np.array，每一个元素在次序表中搜索对应，并用次序表中这个元素的下标
    来取代np.array中该元素的值
    这样就可以把一个string类型的列表转化为一个one-hot编码方式所需求得顺序np.array数组
    '''
    #using os operation to get the number of classes
    print 'the original labels is '
    #print labels
    folders = os.listdir(DataBase_Root_Path)
    folders = np.asarray(folders, dtype = np.float32)
    print folders
    image_num = len(labels)
    print "the total image number is " + str(image_num)
    class_num = len(folders)
    print 'the total class number is ' + str(class_num)
    standard = findMininumLabel(folders)
    print 'the standard index is '
    print standard
    np_array = np.zeros(shape = (class_num,))
    standard_array = np.zeros(shape = (image_num,))

    #old method to get int-type array
    #in face we can just use np.asarray to transform
    #transform from string to int
    count = 0
    for folder in folders:
        temp = int(folder) - standard
        np_array[count] = temp
        count += 1
    print 'the relative class_sheet generated'
    print np_array

    #bubbleSort to get order sheet
    order_array = BubbleSort(np_array)
    print 'order sheet generated'
    print order_array

    #using order sheet to generate one-hot-suited labels
    num = 0
    flag = True
    for label in labels:
        temp = int(label) - standard
        for i in range(class_num):
            if int(label) == 1611472 and flag:
	        print ("zqy now i caught u~")
		flag = False
            if temp == order_array[i]:
                standard_array[num] = i
                num += 1

    return standard_array


def create_label(path):
    #using the student ID as th label
    for File in os.listdir(path):
        #find all files(include foldes) in the current directory
        full_path = os.path.abspath(os.path.join(path,File))
        if os.path.isdir(full_path):
            create_label(full_path)
        else:
            if File.endswith('.png') or File.endswith('.PNG'):
                #getting the upper level path
                folder_path = os.path.abspath(os.path.join((full_path),"../"))
                upper_folder_path = os.path.abspath(os.path.dirname(folder_path))
                #do a string substraction
                current_label = getLastPath(folder_path, upper_folder_path)
                labels.append(current_label)
    #print labels
    return labels

def find_image(path):
    #recursion to find the data
    for File in os.listdir(path):
        #find all files(include foldes) in the current directory
        full_path = os.path.abspath(os.path.join(path,File))
        if os.path.isdir(full_path):
            find_image(full_path)
        else:
            if File.endswith('.png') or File.endswith('.PNG'):
                image = cv2.imread(full_path)
                #Uncomment this part to see current image
                #but may cause system lag
                #cv2.imshow('current_image', image)
                image = resize_image(image)
                images.append(image)
    return images


def load_dataset(path):
    np.set_printoptions(threshold = 1e6)
    labels = create_label(path)
    float_labels = np.asarray(labels, dtype = np.float32)
    #print '=================================='
    #print 'this is a temporary float32-type label sheet'
    #print float_labels
    #standardlize all labels
    #print 'original labels is'
    #print labels
    res = labelStandardlization(float_labels)
    print 'one-hot-suited labels generated'

    #load DataBase images
    images = find_image(path)
    #convert to numpy matrix
    images = np.array(images)

    print '==========================='
    print '||image array shape is : ||'
    print images.shape
    print '==========================='
    #print res
    print '||labels array shape is :||'
    print res.shape
    print '==========================='

    return images,res

if __name__ == '__main__':
    images,labels = load_dataset(DataBase_Root_Path)

    '''
    #One Hot code test here
    labels = np_utils.to_categorical(labels, 5)
    print labels
    '''
