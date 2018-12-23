# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
Date: 2018/11/21
Author: Xu Yucheng 1611453
Abstract: Code for loading dataset and spliting for train_data, valid_data, test_data 
'''
import os
import sys
import numpy as np
import cv2
import timeit
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from utils import clock, resize_image, Bubble_Sort, Get_ID

TESTDATA_DIR = '/home/kamerider/machine_learning/face_recognition/TestData'

class Dataset:
    def __init__(self, path):
        self.DATASET_ROOT_DIR = path
        self.IMAG_SIZE = 224
        self.minimun_id = 9999999
        #order_sheet to generate one-hot sheet
        self.order_sheet = np.array ([], dtype=np.int32)
        self.relative_sheet = np.array ([], dtype=np.uint32)

        #self.images = np.array ([], dtype=np.float32)
        self.images = []
        self.labels = np.array ([], dtype=np.uint8)
        #number of predefined classes
        self.predefined_class = None
        
        #test_set
        self.test_image = []
        self.test_label = np.array ([], dtype=np.uint8)

        self.input_shape = (128,128,3)
    
    def Generate_OrderSheet (self):
        '''
        初始得到的标签列表元素为string类型
        首先将其转化为np.array
        然后根据得到的np.array中元素的大小，从小到大排序得到一张次序表
        然后遍历np.array，每一个元素在次序表中搜索对应，并用次序表中这个元素的下标
        来取代np.array中该元素的值
        这样就可以把一个string类型的列表转化为一个one-hot编码方式所需求得顺序np.array数组
        '''
        #using os operation to get the number of classes
        print 'the original class labels is: '
        #print labels
        folders = os.listdir (self.DATASET_ROOT_DIR)
        folders = np.asarray (folders, dtype = np.float32)
        print folders

        self.minimun_id = min(folders)
        print 'the minimum class label is: ' + str(self.minimun_id)

        class_num = len(folders)
        self.predefined_class = class_num
        print 'the total number of predefined classes is: ' + str(class_num)

        for folder in folders:
            temp = int(folder) - self.minimun_id
            self.relative_sheet = np.append (self.relative_sheet, temp)
        print 'the relative class label sheet is: '
        print self.relative_sheet

        self.order_sheet = Bubble_Sort (self.relative_sheet)
        print 'generated order sheet: '
        print self.order_sheet
    
    @clock
    def Find_Image (self, path):
        for File in os.listdir (path):
            full_path = os.path.abspath (os.path.join(path,File))
            if os.path.isdir (full_path):
                self.Find_Image (full_path)
            else:
                if File.endswith('.png') or File.endswith('.PNG'):

                    #Find & Store images
                    image = cv2.cvtColor (
                        cv2.imread(full_path), cv2.COLOR_BGR2RGB
                    ).astype(np.float32)/255
                    if image.shape != (128,128,3):
                        print 'detected image with incorrect size: ' + full_path
                        print 'the shape of this image is:' + str(image.shape)
                        image = resize_image (image)
                    #image = cv2.imread(full_path)
                    self.images.append(image)

                    #Create Labels
                    #getting the upper level path
                    folder_path = os.path.abspath(os.path.join((full_path),"../.."))
                    upper_folder_path = os.path.abspath(os.path.dirname(folder_path))
                    #do a string substraction
                    current_label = Get_ID(folder_path, upper_folder_path)

                    for i in range(len(self.order_sheet)):
                        temp = int(current_label) - self.minimun_id
                        if temp == self.order_sheet[i]:
                            current_label = i
                    
                    self.labels = np.append (self.labels, current_label)

    @clock
    def Test_Data (self, path):
        for File in os.listdir (path):
            full_path = os.path.abspath (os.path.join(path,File))
            if os.path.isdir (full_path):
                self.Test_Data(full_path)
            else:
                if File.endswith('.png') or File.endswith('.PNG'):

                    #Find & Store images
                    image = cv2.cvtColor (
                        cv2.imread(full_path), cv2.COLOR_BGR2RGB
                    ).astype(np.float32)/255
                    if image.shape != (128,128,3):
                        print 'detected image with incorrect size: ' + full_path
                        print 'the shape of this image is:' + str(image.shape)
                        image = resize_image (image)
                    #image = cv2.imread(full_path)
                    self.test_image.append(image)

                    #Create Labels
                    #getting the upper level path
                    folder_path = os.path.abspath(os.path.join((full_path),"../"))
                    upper_folder_path = os.path.abspath(os.path.dirname(folder_path))
                    #do a string substraction
                    current_label = Get_ID(folder_path, upper_folder_path)

                    for i in range(len(self.order_sheet)):
                        temp = int(current_label) - self.minimun_id
                        if temp == self.order_sheet[i]:
                            current_label = i
                    
                    self.test_label = np.append (self.labels, current_label)

    
    def Load_Dataset (self):
        #set maximum output in terminal
        np.set_printoptions(threshold = 1e6)
        print 'Generating Order Sheet'
        self.Generate_OrderSheet()
        print 'Find All images & Generate Labels'
        self.Find_Image(self.DATASET_ROOT_DIR)
        self.Test_Data(TESTDATA_DIR)

        self.images = np.array(self.images)
        self.test_image = np.array(self.test_image)

        print '==========================='
        print '||image array shape is : ||'
        print '==========================='
        print self.images.shape
        print '==========================='
        print '||labels array shape is :||'
        print '==========================='
        print self.labels.shape
        print '==========================='   
        print '||test image shape is :  ||'
        print '==========================='
        print self.test_image.shape
        print '==========================='    

       
if __name__ == '__main__':
    dataset = Dataset(sys.argv[1])
    dataset.Load_Dataset()


        

        






        

        
        






