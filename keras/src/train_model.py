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
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras import backend as K

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
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
#early stopping callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

#keras wrapper for scikit-learn
from keras.wrappers.scikit_learn import KerasClassifier


from utils import *
from load_data import Dataset
from creat_model import Model

# Uncomment to use a small dataset for test
#DATABASE_ROOT_DIR = '/home/kamerider/Documents/small_dataset'
DATABASE_ROOT_DIR = '/home/kamerider/Documents/DataBase'
MODEL_PATH = '../model/'
CHECKPOINT_DIR = '../checkpoint'
LOG_DIR = '../tensorboard'

class Train:
    #在此处修改batch_size, nb_epoch, data_augmentation
    def __init__(self, model, batch_size=64, nb_epoch=200, data_augmentation=False):
        #train set
        self.train_image = None
        self.train_label = None
        
        #valid set
        self.valid_image = None
        self.valid_label = None

        #test set
        self.test_image = None
        self.test_label = None

        #KFold cross validation
        self.kfold = None
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.data_augmentation = data_augmentation
        self.VGG_16 = model

        #training output
        self.loss = []
        self.acc = []
        self.callbacks=[]
    
    def convert2oneHot(self, labels, class_num):
        onehot_label = np_utils.to_categorical(labels, class_num)
        return onehot_label
    
    def setCallbacks(self, option):
        

        global CHECKPOINT_DIR
        CHECKPOINT_DIR = os.path.abspath(CHECKPOINT_DIR)
        if option == 'DataShuffleSplit':
            if not self.data_augmentation:
                CHECKPOINT_DIR = os.path.abspath(os.path.join(CHECKPOINT_DIR, "ShuffleSplit"))
            else:
                CHECKPOINT_DIR = os.path.abspath(os.path.join(CHECKPOINT_DIR, "ShuffleSplit_augmentation"))

        if option == 'KFoldM':
            CHECKPOINT_DIR = os.path.abspath(os.path.join(CHECKPOINT_DIR, "KFold_manual"))

        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)

        filename = "model_{epoch:02d}-{val_acc:.2f}.h5"
        CHECK_POINT = os.path.abspath(os.path.join(CHECKPOINT_DIR, filename))

        #early stopping
        #10次迭代val_loss不下降则停止训练
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

        #使用CheckPoint来记录训练过程
        checkpoint = ModelCheckpoint(
            CHECK_POINT, monitor='val_acc', verbose=1, 
            save_best_only=True, save_weights_only=False,
            mode='auto',period=10
            )
        #使用ReduceLROnPlateau在val_loss不再下降的时候降低学习率
        reducelr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, 
            patience=10, verbose=0, mode='auto', 
            epsilon=0.0001, cooldown=0, min_lr=0
            )
        global LOG_DIR
        #使用Tensorboard可视化训练过程
        tensorboard = TensorBoard(
            log_dir='LOG_DIR', histogram_freq=0, batch_size=64, write_graph=True, 
            write_grads=False, write_images=False, embeddings_freq=0, 
            embeddings_layer_names=None, embeddings_metadata=None
            )

        self.callbacks = [tensorboard, early_stopping, checkpoint, reducelr]


    #使用ShuffleSplit划分数据集为训练集和测试集进行训练
    #比例为7：3
    def train_with_SplitedData(self, dataset):

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
        self.test_image = dataset.test_image

        self.test_label = self.convert2oneHot(dataset.test_label, dataset.predefined_class)
        self.train_label = self.convert2oneHot(self.train_label, dataset.predefined_class)
        self.valid_label = self.convert2oneHot(self.valid_label, dataset.predefined_class)
        print (" Size of Train_Image is: ", self.train_image.shape)
        print (" Size of Valid_Image is: ", self.valid_label.shape)

        #利用划分好的训练集和数据集进行训练
        print ('Using ShuffleSplit to split Dataset')
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
            #define callback funcs
            #在使用ShuffleSplit划分训练集的时候使用EarlyStopping来检测
            #val_loss是否不再下降，也可以防止过拟合现象

            #recording loss, loss_val, accuracy, accuracy_val
            #and wirte it to Train_History.txt
            hist = self.VGG_16.model.fit(self.train_image,
                           self.train_label,
                           batch_size = self.batch_size,
                           nb_epoch = self.nb_epoch,
                           #validation_split = 0.33,
                           validation_data=(self.valid_image, self.valid_label),
                           #验证集是用来在训练过程中优化参数的，可以直接使用validation_split从训练集中划分出来
                           shuffle = True,
                           callbacks=self.callbacks)

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

            #define callback funcs
            #在使用ShuffleSplit划分训练集的时候使用EarlyStopping来检测
            #val_loss是否不再下降，也可以防止过拟合现象

            #利用生成器开始训练模型
            hist = self.VGG_16.model.fit_generator(datagen.flow(self.train_image, self.train_label,
                                                   batch_size = self.batch_size),
                                     samples_per_epoch = self.train_image.shape[0],
                                     nb_epoch = self.nb_epoch,
                                     validation_data = (self.valid_image, self.valid_label),
                                     callbacks=self.callbacks)
    
    #Using KFold Cross Validate
    #Warning! This may take long time to train
    #此处根据KFold划分的结果，手动利用每次划分出来的数据进行训练
    def train_with_CrossValidation_Manual(self, dataset):
        #if you want to use Scikit_learn wrapper for Keras
            #please uncomment this function
            '''
            self.train_with_CrossValidation(dataset)
            '''
            #and comment the following lines

            #默认使用人工划分交叉验证，并使用每次划分出来的数据进行训练和验证
            #采用KFold方法划分训练集
            seed = 7
            np.random.seed(seed)
            print 
            dataset.images, dataset.labels = shuffle(dataset.images, dataset.labels, random_state=seed)
            self.kfold = StratifiedKFold (n_splits=10, shuffle=True, random_state=seed)
            print (self.kfold)
            print (self.kfold.split(dataset.images, dataset.labels))
            Cross_Validate_Scores=[]
            num = 1 
            for train_idx, valid_idx in self.kfold.split(dataset.images, dataset.labels):
                #每一个fold的checkpoint放在一个文件夹中
                foldername = "Fold_"+str(num)
                CHECKPOINT_DIR = os.path.abspath(os.path.join(CHECK_POINT, foldername))
                if not os.path.exists(CHECKPOINT_DIR):
                    os.makedirs(CHECKPOINT_DIR)
                CHECK_POINT = os.path.join(CHECKPOINT_DIR, filename)

                print ("\033[0;31;40mFold %d/10 start\033[0m"%(num))
                num +=1 
                self.train_image = dataset.images[train_idx]
                self.train_label = dataset.labels[train_idx]
                self.valid_image = dataset.images[valid_idx]
                self.valid_label = dataset.labels[valid_idx]

                self.test_image = dataset.test_image
                self.test_label = dataset.test_label

                #convert to one-hot code
                self.train_label = self.convert2oneHot(self.train_label, dataset.predefined_class)
                self.valid_label = self.convert2oneHot(self.valid_label, dataset.predefined_class)
                self.test_label  = self.convert2oneHot(self.test_label, dataset.predefined_class)
                print ("\033[0;31;40m Size of Train_Image is: \033[0m", self.train_image.shape)
                print ("\033[0;31;40m Size of Valid_Image is: \033[0m", self.valid_image.shape)
                
                #compile model
                self.VGG_16.model.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])   #完成实际的模型配置工作

                hist = self.VGG_16.model.fit(
                    self.train_image,self.train_label,
                    batch_size = self.batch_size,
                    nb_epoch = self.nb_epoch,
                    #validation_split = 0.33,
                    validation_data=(self.valid_image, self.valid_label),
                    #验证集是用来在训练过程中优化参数的，可以直接使用validation_split从训练集中划分出来
                    shuffle = True,
                    callbacks=self.callbacks
                    )
                self.loss.extend(hist.history['loss'])
                self.acc.extend(hist.history['acc'])
                scores = self.VGG_16.model.evaluate(self.test_image, self.test_label, verbose=1)

                print ('[INFO] %s: %.2f%%'%(self.VGG_16.model.metrics_names[1],scores[1]*100))
                Cross_Validate_Scores.append(scores[1]*100)
            Cross_Validate_Scores = np.asarray(Cross_Validate_Scores)
            print ('[INFO] %.2f%% (+/- %.2f%%)'%(np.mean(Cross_Validate_Scores), np.std(Cross_Validate_Scores)))

    #Using KFold Cross Validate
    #Warning! This may take long time to train
    #由于keras并没有实现交叉检验的函数，所以这里通过使用keras中scikit-learn的api来实现cross validation
    def train_with_CrossValidation_Wrapper(self, dataset):
        #CrossValidation use scikit-learn wrapper
        estimator = KerasClassifier(
            build_fn=self.VGG_16.model_for_scikit,
            nb_epoch=self.nb_epoch,
            batch_size=self.batch_size,
            verbose=1
        )
        train_x = dataset.images
        train_y = np_utils.to_categorical(dataset.labels, dataset.predefined_class)
        estimator.fit(train_x, train_y)

        seed = 23
        np.random.seed(seed)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        results = cross_val_score(estimator, train_x, dataset.labels, cv=kfold)
        print ('[INFO] %.2f%% (+/- %.2f%%)'%(np.mean(results)*100.0, np.std(results)*100.0))

    #使用GridSearchCV的方法来选择模型的最优超参数
    def train_with_GridSearchCV(self, dataset):
        estimator = KerasClassifier(
            build_fn=self.VGG_16.model_for_scikit,
            verbose=1
        )#一个scikit_learn的warapper

        optimizers = ['rmsprop', 'adam', 'sgd'] #优化器
        init_modes = ['glorot_uniform', 'normal', 'uniform'] #初始化模式
        nb_epoch = np.array([25, 50, 100, 150]) #Epoch数
        batch_size = np.array([16, 32, 64]) #batch_size

        # 网格字典optimizer和init_mode是模型的参数，epochs和batch_size是wrapper的参数
        param_grid = dict(optimizer=optimizers, epochs=nb_epoch, batch_size=batch_size, init_mode=init_modes)
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=4)

        train_x = dataset.images
        train_y = np_utils.to_categorical(dataset.labels, dataset.predefined_class)
        grid_result = grid.fit(train_x, train_y)

        print ("[INFO] Best: %f using %s"%(grid_result.best_score_, grid_result.best_params_))
        for params, mean_score, scores in grid_result.grid_scores_:
            print('[INFO] %f (%f) with %r' % (scores.mean(), scores.std(), params))

def train_model_main(option):
    #load dataset
    dataset = Dataset (DATABASE_ROOT_DIR)
    dataset.Load_Dataset()
    #build model
    VGG_16 = Model()
    VGG_16.build_model(dataset) 
    train_vgg = Train(VGG_16)

    train_vgg.setCallbacks(option)
    global MODEL_PATH
    if option == 'DataShuffleSplit':
        MODEL_PATH = os.path.abspath(os.path.join(MODEL_PATH, "ShuffleSplit_model.h5"))
        train_vgg.train_with_SplitedData(dataset)
        VGG_16.evaluate_model(train_vgg.test_image, train_vgg.test_label)
        VGG_16.save_model(MODEL_PATH)
    
    elif option == 'KFoldM':
        MODEL_PATH = os.path.abspath(os.path.join(MODEL_PATH, "KFold_Manual_model.h5"))
        train_vgg.train_with_CrossValidation_Manual(dataset)
        VGG_16.save_model(MODEL_PATH)

    elif option == 'KFoldW':
        MODEL_PATH = os.path.abspath(os.path.join(MODEL_PATH, "KFold_Wrapper_model.h5"))
        train_vgg.train_with_CrossValidation_Wrapper(dataset)
        VGG_16.save_model(MODEL_PATH)

    elif option == 'GridSearch':
        print ("[WARNING!!!] THIS mode is not available!")
        exit(0)
        train_vgg.train_with_GridSearchCV(dataset)
    
    elif option == 'help':
        print_usage(sys.argv[0])
    
    else:
        print_usage(sys.argv[0])




if __name__ == '__main__':
    numOfargv = len(sys.argv)
    if numOfargv < 2:
        print_usage(sys.argv[0])
        
    option = sys.argv[1]
    train_model_main(option)
    
    

        


        