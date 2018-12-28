# -*- coding: utf-8 -*-
#!/usr/bin/env python
#use random to generate train set, test set, valid set
'''
Date: 2018/11/23
Author: Xu Yucheng
Abstract: Code for spliting data, training model
'''
import random

#Tensorflow part & Keras part
import os
import sys
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from keras import backend as K
from load_data import Dataset
from creat_model import Model

#Tensorflow part & Keras part
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
from keras import backend as K

#DATABASE_ROOT_DIR = '/home/kamerider/machine_learning/face_recognition/DataBase'
DATABASE_ROOT_DIR = '/home/kamerider/Documents/DataBase'
MODEL_PATH = '/home/kamerider/machine_learning/face_recognition/model/face_recognition.h5'
HISTORY_PATH = '/home/kamerider/machine_learning/face_recognition/History/Train_History.txt'
FIGURE_PATH = '/home/kamerider/machine_learning/face_recognition/History'



class Train:
    def __init__(self, model, batch_size=64, nb_epoch=10, data_augmentation=False):
        #train set
        self.train_image = None
        self.train_label = None
        
        #valid set
        self.valid_image = None
        self.valid_label = None

        #KFold cross validation
        self.kfold = None
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.data_augmentation = data_augmentation
        self.VGG_16 = model
    
    def convert2oneHot(self, labels, class_num):
        onehot_label = np_utils.to_categorical(labels, class_num)
        return onehot_label


    def split_dataset(self, dataset, options):
        if options == '-KFold':
            #采用KFold方法划分训练集
            seed = 7
            np.random.seed(seed)
            print 
            dataset.images, dataset.labels = shuffle(dataset.images, dataset.labels, random_state=seed)
            self.kfold = StratifiedKFold (n_splits=4, shuffle=True, random_state=seed)
            print (self.kfold)
            print (self.kfold.split(dataset.images, dataset.labels))
            for train_idx, valid_idx in self.kfold.split(dataset.images, dataset.labels):
                self.train_image = dataset.images[train_idx]
                self.train_label = dataset.labels[train_idx]
                self.valid_image = dataset.images[valid_idx]
                self.valid_label = dataset.labels[valid_idx]
                #convert to one-hot code
                self.train_label = self.convert2oneHot(self.train_label, dataset.predefined_class)
                self.valid_label = self.convert2oneHot(self.valid_label, dataset.predefined_class)
                print ("\033[0;31;40m Size of Train_Image is: \033[0m", self.train_image.shape)
                print ("\033[0;31;40m Size of Valid_Image is: \033[0m", self.valid_image.shape)

        if options == '-Split':
            #手动划分训练集和验证集
            seed = 7
            np.random.seed(seed)
            train_idx, val_idx = next(iter(
                StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed).split(dataset.images, dataset.labels)
            ))
            self.train_image = dataset.images[train_idx]
            self.train_label = dataset.labels[train_idx]
            self.valid_image = dataset.images[val_idx]
            self.valid_label = dataset.labels[val_idx]
            self.train_label = self.convert2oneHot(self.train_label, dataset.predefined_class)
            self.valid_label = self.convert2oneHot(self.valid_label, dataset.predefined_class)
            print (" Size of Train_Image is: ", self.train_image.shape)
            print (" Size of Valid_Image is: ", self.valid_label.shape)

    def train(self):

        #optimizers
        sgd = SGD(lr = 0.01, decay = 1e-6,
                  momentum = 0.9, nesterov = True) #采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        #compile model
        self.VGG_16.model.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])   #完成实际的模型配置工作

        #不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        #训练数据，有意识的提升训练数据规模，增加模型训练量
        if not self.data_augmentation:
            #recording loss, loss_val, accuracy, accuracy_val
            #and wirte it to Train_History.txt
            hist = self.VGG_16.model.fit(self.train_image,
                           self.train_label,
                           batch_size = self.batch_size,
                           nb_epoch = self.nb_epoch,
                           #validation_split = 0.33,
                           validation_data=(self.valid_image, self.valid_label),
                           #验证集是用来在训练过程中优化参数的，可以直接使用validation_split从训练集中划分出来
                           shuffle = True)
            with open(HISTORY_PATH,'w') as f:
                f.write(str(hist.history))

            #visualization
            #store the output history
            model_val_loss = hist.history['val_loss']
            model_val_acc  = hist.history['val_acc']
            model_loss     = hist.history['loss']
            model_acc      = hist.history['acc']

            #Using matplotlib to visualize
            epochs = np.arange(self.nb_epoch)+1
            plt.figure()
            plt.plot(epochs, model_val_loss, label = 'model_val_loss')
            plt.plot(epochs, model_loss, label = 'model_loss')
            plt.title('visualize the training process')
            plt.xlabel('Epoch #')
            plt.ylabel('Validation Loss & Train Loss')
            plt.legend()
            plt.savefig(FIGURE_PATH+'/loss_figure.png')
            plt.show()

            plt.figure()
            plt.plot(epochs, model_val_acc, label = 'model_val_acc')
            plt.plot(epochs, model_acc, label = 'model_acc')
            plt.title('visualize the training process')
            plt.xlabel('Epoch #')
            plt.ylabel('Validation accuracy & Train accuracy')
            plt.legend()
            plt.savefig(FIGURE_PATH+'/acc_figure.png')
            plt.show()

        #使用实时数据提升
        else:
            #定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            #次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center = False,             #是否使输入数据去中心化（均值为0），
                samplewise_center  = False,             #是否使输入数据的每个样本均值为0
                featurewise_std_normalization = False,  #是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization  = False,  #是否将每个样本数据除以自身的标准差
                zca_whitening = False,                  #是否对输入数据施以ZCA白化
                rotation_range = 20,                    #数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range  = 0.2,               #数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range = 0.2,               #同上，只不过这里是垂直
                horizontal_flip = True,                 #是否进行随机水平翻转
                vertical_flip = False)                  #是否进行随机垂直翻转

            #计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(self.train_image)

            #利用生成器开始训练模型
            hist = self.VGG_16.model.fit_generator(datagen.flow(self.train_image, self.train_label,
                                                   batch_size = self.batch_size),
                                     samples_per_epoch = self.train_image.shape[0],
                                     nb_epoch = self.nb_epoch,
                                     validation_data = (self.valid_image, self.valid_label))
            with open(HISTORY_PATH,'w') as f:
                f.write(str(hist.history))

            #visualization
            #store the output history
            model_val_loss = hist.history['val_loss']
            model_val_acc  = hist.history['val_acc']
            model_loss     = hist.history['loss']
            model_acc      = hist.history['acc']

            #Using matplotlib to visualize
            epochs = np.arange(self.nb_epoch)+1
            plt.figure()
            plt.plot(epochs, model_val_loss, label = 'model_val_loss')
            plt.plot(epochs, model_loss, label = 'model_loss')
            plt.title('visualize the training process')
            plt.xlabel('Epoch #')
            plt.ylabel('Validation Loss & Train Loss')
            plt.legend()
            plt.savefig(FIGURE_PATH+'/loss_figure.png')
            plt.show()

            plt.figure()
            plt.plot(epochs, model_val_acc, label = 'model_val_acc')
            plt.plot(epochs, model_acc, label = 'model_acc')
            plt.title('visualize the training process')
            plt.xlabel('Epoch #')
            plt.ylabel('Validation accuracy & Train accuracy')
            plt.legend()
            plt.savefig(FIGURE_PATH+'/acc_figure.png')
            plt.show()

if __name__ == '__main__':
    dataset = Dataset (DATABASE_ROOT_DIR)
    dataset.Load_Dataset()
    VGG_16 = Model()
    VGG_16.build_model(dataset)
    train_vgg = Train(VGG_16)

    '''
    @TODO
    从控制台读取训练参数(batch_size, nb_epoch, data_augmentation)
    '''
    
    options = ['-Split']
    for option in options:
        print (option, "data split result is: ")
        train_vgg.split_dataset(dataset, option)
        train_vgg.train()
        VGG_16.save_model()

        


        