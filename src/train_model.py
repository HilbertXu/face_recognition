# -*- coding: utf-8 -*-
#!/usr/bin/env python
#use random to generate train set, test set, valid set
import random

#Tensorflow part & Keras part
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras import optimizers
from keras.utils import np_utils
from keras.models import load_model

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from keras import backend as K
from load_data import Dataset

#DATABASE_ROOT_DIR = '/home/kamerider/machine_learning/face_recognition/DataBase'
DATABASE_ROOT_DIR = '/home/kamerider/Documents/DataBase'

class Train:
    def __init__(self):
        #train set
        self.train_image = None
        self.train_label = None
        
        #valid set
        self.valid_image = None
        self.valid_label = None

        #KFold cross validation
        self.kfold = None
        self.input_shape = (128,128,3)

    def split_dataset(self, dataset, options):
        if options == '-KFold':
            #采用KFold方法划分训练集
            seed = 7
            np.random.seed(seed)
            print 
            dataset.images, dataset.labels = shuffle(dataset.images, dataset.labels, random_state=seed)
            self.kfold = StratifiedKFold (n_splits=10, shuffle=True, random_state=seed)
            print self.kfold
            print self.kfold.split(dataset.images, dataset.labels)
            #for train_idx, valid_idx in self.kfold.split(dataset.images, dataset.labels):
                #print ("Train_Index: ", train_idx, "size of train index is: ", train_idx.shape)
                #print ("Valid_Index: ", valid_idx, "size of valid index is: ", valid_idx.shape)

        if options == '-Split':
            #手动划分训练集和验证集
            seed = 7
            np.random.seed(seed)
            train_idx, val_idx = next(iter(
                StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed).split(dataset.images, dataset.labels)
            ))
            print ("Train_Index size is: ", train_idx.shape)
            print ("valid_Index size is: ", val_idx.shape)
            self.train_image = dataset.images[train_idx]
            self.train_label = dataset.labels[train_idx]
            self.valid_image = dataset.images[val_idx]
            self.valid_label = dataset.labels[val_idx]

if __name__ == '__main__':
    train = Train()
    dataset = Dataset (DATABASE_ROOT_DIR)
    dataset.Load_Dataset()

    options = ['-KFold', '-Split']
    for option in options:
        print (option, "data split result is: ")
        train.split_dataset(dataset, option)
        


        