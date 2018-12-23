# -*- coding: utf-8 -*-
#!/usr/bin/env python

#use random to generate train set, test set, valid set
import random

#Tensorflow part & Keras part
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

#load the predifined function to load all the images to dataset
from loadFaceDataset import load_dataset, IMAGE_SIZE, DataBase_Root_Path
#The ROOT path of DataSet

MODEL_PATH = '/home/kamerider/catkin_ws/src/machine_vision/model/face_detect.h5'
HISTORY_PATH = '/home/kamerider/catkin_ws/src/machine_vision/History/Train_History.txt'

class DataSet:
    def __init__(self, path_name):
        #train set
        self.train_images = None
        self.train_labels = None

        #valid set
        self.valid_images = None
        self.valid_labels = None

        #test set
        self.test_images  = None
        self.test_labels  = None

        #ROOT path of DataSet
        self.path = path_name

        #Input shape default
        self.input_shape = None

    def load(self, image_rows = IMAGE_SIZE, image_cols = IMAGE_SIZE,
        image_channels = 3, predifined_class = 5):
        #change the number of "predifined_class" here to support larger DataSet

        #load images & labels
        images, labels = load_dataset(self.path)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images,labels,
                                                                                       test_size = 0.3, random_state = random.randint(10000,10010))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size = 0.5, random_state =
                                                               random.randint(10000,10010))
        '''
        gedit ~/.keras/keras.json
        file details:
        {
            "epsilon": 1e-07,
            "floatx": "float32",
            "image_data_format": "channels_last",
            "backend": "tensorflow"
        }
        make sure the dimesion order of images is as same as Keras backend required
        tensorflow requires channels_last
        theano requires channel_first
        '''
        #if using Theano as backend make a transform for every image
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], image_channels, image_rows, image_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], image_channels, image_rows, image_cols)
            test_images = test_images.reshape(test_images.shape[0], image_channels, image_rows, image_cols)
            self.input_shape = (image_channels, image_rows, image_cols)


        if K.image_dim_ordering() == 'tf':
            train_images = train_images.reshape(train_images.shape[0], image_rows, image_cols, image_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], image_rows, image_cols, image_channels)
            test_images = test_images.reshape(test_images.shape[0], image_rows, image_cols, image_channels)
            self.input_shape = (image_rows, image_cols, image_channels)

        print (train_images.shape[0], 'train samples')
        print (valid_images.shape[0], 'valid samples')
        print (test_images.shape[0], 'test samples')

        #using One_hot codeing to deal with our labels and transform it to a vector
        #using categorical_crossentropy as loss-function
        train_labels =np_utils.to_categorical(train_labels, predifined_class)
        valid_labels =np_utils.to_categorical(valid_labels, predifined_class)
        test_labels =np_utils.to_categorical(test_labels, predifined_class)

        #floatilize the pixel
        train_iamges = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images  = test_images.astype('float32')

        #image normalization
        train_images /= 255
        valid_images /= 255
        test_images  /= 255

        #assign the self varibles
        self.train_images = train_iamges
        self.valid_images = valid_images
        self.test_images  = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels  = test_labels


class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes = 5):
        #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential()

        #以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                     input_shape = dataset.input_shape))    #1 2维卷积层
        self.model.add(Activation('relu'))                                  #2 激活函数层

        self.model.add(Convolution2D(32, 3, 3))                             #3 2维卷积层
        self.model.add(Activation('relu'))                                  #4 激活函数层

        self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #5 池化层
        self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
        self.model.add(Dropout(0.25))                                       #6 Dropout层

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))         #7  2维卷积层
        self.model.add(Activation('relu'))                                  #8  激活函数层

        self.model.add(Convolution2D(64, 3, 3))                             #9  2维卷积层
        self.model.add(Activation('relu'))                                  #10 激活函数层

        self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #11 池化层
        self.model.add(Dropout(0.25))                                       #12 Dropout层

        self.model.add(Flatten())                                           #13 Flatten层
        self.model.add(Dense(512))                                          #14 Dense层,又被称作全连接层
        self.model.add(Activation('relu'))                                  #15 激活函数层
        self.model.add(Dropout(0.5))                                        #16 Dropout层
        self.model.add(Dense(nb_classes))                                   #17 Dense层
        self.model.add(Activation('softmax'))                               #18 分类层，输出最终结果

        #输出模型概况
        self.model.summary()
    #训练模型
    def train(self, dataset, batch_size = 64, nb_epoch = 200, data_augmentation = False):
        sgd = SGD(lr = 0.01, decay = 1e-6,
                  momentum = 0.9, nesterov = True) #采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])   #完成实际的模型配置工作

        #不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        #训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:
            #recording loss, loss_val, accuracy, accuracy_val
            #and wirte it to Train_History.txt
            hist = self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size = batch_size,
                           nb_epoch = nb_epoch,
                           validation_data = (dataset.valid_images, dataset.valid_labels),
                           shuffle = True)
            with open(HISTORY_PATH,'w') as f:
                f.write(str(hist.history))

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
            datagen.fit(dataset.train_images)

            #利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                   batch_size = batch_size),
                                     samples_per_epoch = dataset.train_images.shape[0],
                                     nb_epoch = nb_epoch,
                                     validation_data = (dataset.valid_images, dataset.valid_labels))
    #识别人脸
    def face_predict(self, image):
        #依然是根据后端系统确定维度顺序
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            #image = resize_image(image)                             #尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))   #与模型训练不同，这次只是针对1张图片进行预测
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            #image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

        #浮点并归一化
        image = image.astype('float32')
        image /= 255

        #给出输入属于各个类别的概率
        result = self.model.predict_proba(image)
        print('result:', result)

        #给出类别预测：每一个类别的置信度
        result = self.model.predict_classes(image)

        #返回类别预测结果
        return result[0]


    def save_model(self, file_path = MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path = MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dateset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

if __name__ == '__main__':
    dataset = DataSet(DataBase_Root_Path)
    dataset.load()

    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(MODEL_PATH)

    model = Model()
    model.load_model(MODEL_PATH)
    model.evaluate(dataset)
