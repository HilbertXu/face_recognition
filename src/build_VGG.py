# -*- coding: utf-8 -*-
#!/usr/bin/env python

#use random to generate train set, test set, valid set
import random

#Tensorflow part & Keras part
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

def build_model(self, dataset, nb_classes = 5):
    #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
    self.model = Sequential()
    self.model.add(Convolution2D(64,4,4, border_mode = 'valid', activation='relu', input_shape = dataset.input_shape))
    self.model.add(Convolution2D(64,3,3, border_mode = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    self.model.add(Dropout(0.3))
    self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Convolution2D(64,4,4, border_mode = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    self.model.add(Convolution2D(64,3,3, border_mode = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    self.model.add(Dropout(0.3))
    self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Convolution2D(128,3,3, border_mode = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.03)))
    self.model.add(Dropout(0.4))
    self,model.add(Convolution2D(128,3,3, border_mode = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.03)))
    self.model.add(Dropout(0.4))
    self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Flatten())
    self.model.add(Dense(512, init='normal',W_regularizer=regularizers.l2(0.02), activity_regularizer=regularizers.l2(0.01)))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.3))

    self.model.add(Dense(512, init='normal',W_regularizer=regularizers.l2(0.02), activity_regularizer=regularizers.l2(0.01)))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.3))

    self.model.add(Dense(nb_classes, init='normal'))
    self.model.add(Activation('softmax'))

    '''
    tested network as fellow
    #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
    self.model = Sequential()
    self.model.add(Convolution2D(32,4,4, border_mode = 'valid', activation='relu', input_shape = dataset.input_shape))
    self.model.add(Convolution2D(32,3,3, border_mode = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    self.model.add(Dropout(0.3))
    self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Convolution2D(32,4,4, border_mode = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    self.model.add(Convolution2D(32,3,3, border_mode = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    self.model.add(Dropout(0.3))
    self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))


    self.model.add(Convolution2D(64,3,3, border_mode = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.03)))
    self.model.add(Dropout(0.4))
    self.model.add(Convolution2D(64,3,3, border_mode = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.03)))
    self.model.add(Dropout(0.4))
    self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Flatten())
    self.model.add(Dense(512, init='normal',W_regularizer=regularizers.l2(0.02), activity_regularizer=regularizers.l2(0.01)))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.3))

    self.model.add(Dense(512, init='normal',W_regularizer=regularizers.l2(0.02), activity_regularizer=regularizers.l2(0.01)))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.3))

    self.model.add(Dense(nb_classes, init='normal'))
    self.model.add(Activation('softmax'))
    #输出模型概况
    self.model.summary()
    '''
    '''
    #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
    self.model = Sequential()
    #以下代码将顺序添加VGG网络需要的各层，一个add就是一个网络层
    self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                 input_shape = dataset.input_shape))    #1 2维卷积层
    self.model.add(Activation('relu'))                                  #2 激活函数层
    self.model.add(Convolution2D(32, 3, 3, border_mode='same', kernel_regularizer=regularizers.l2(0.02)))                           #3 2维卷积层
    self.model.add(Activation('relu'))                                  #4 激活函数层

    self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9,
                                        weights=None, beta_init='zero', gamma_init='one'))             #6 BN层
    self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #7 池化层
    self.model.add(Dropout(0.3))                                        #5 Dropout层

    self.model.add(Convolution2D(64, 4, 4, border_mode='same', kernel_regularizer=regularizers.l2(0.02)))         #7  2维卷积层
    self.model.add(Activation('relu'))                                  #8  激活函数层
    self.model.add(Convolution2D(64, 3, 3, kernel_regularizer=regularizers.l2(0.02)))                             #9  2维卷积层
    self.model.add(Activation('relu'))                                  #10 激活函数层

    #self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9,
    #                                    weights=None, beta_init='zero', gamma_init='one'))  #12 BN层
    self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #13 池化层
    self.model.add(Dropout(0.25))                                        #11 Dropout层

    self.model.add(Flatten())                                           #14 Flatten层
    self.model.add(Dense(512, init='normal',kernel_regularizer=regularizers.l2(0.02), activity_regularizer=regularizers.l2(0.01)))
    self.model.add(Activation('relu'))                                  #15 激活函数层

    self.model.add(Dropout(0.5))                                        #16 Dropout层
    self.model.add(Dense(nb_classes, init='normal'))                                   #17 Dense层
    self.model.add(Activation('softmax'))                               #18 分类层，输出最终结果

    #输出模型概况
    self.model.summary()
    '''
    '''
    #test another network
    self.model.add(Convolution2D(64, 4, 4, border_mode='valid', input_shape=dataset.input_shape))
    self.model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.3))
    #self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Convolution2D(64, 4, 4, border_mode='valid'))
    self.model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.3))
    #self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Convolution2D(128, 4, 4, border_mode='valid'))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.3))
    #self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))

    self.model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.3))
    #self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))

    self.model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.3))
    #self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Flatten())
    self.model.add(Dense(512, init='normal',W_regularizer=regularizers.l2(0.02), activity_regularizer=regularizers.l2(0.01)))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.3))

    self.model.add(Dense(512, init='normal',W_regularizer=regularizers.l2(0.02), activity_regularizer=regularizers.l2(0.01)))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.3))

    self.model.add(Dense(nb_classes, init='normal'))
    self.model.add(Activation('softmax'))
    '''


    #输出模型概况
    self.model.summary()
