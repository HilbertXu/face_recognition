# -*- coding: utf-8 -*-
#!/usr/bin/env python

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

MODEL_PATH = '/home/kamerider/machine_learning/face_recognition/model/face_recognition.h5'
HISTORY_PATH = '/home/kamerider/machine_learning/face_recognition/History/Train_History.txt'
FIGURE_PATH = '/home/kamerider/machine_learning/face_recognition/History'

class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes = 62):
        #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        #使用了一个标准的VGG-16网络结构
        #输入之前需要将图像resize成为64x64的大小
        self.model = Sequential()

        #以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        self.model.add(Convolution2D(64, 3, 3, border_mode='same',
                                     input_shape = dataset.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        #对于BN层的参数，如果如果输入是形如（samples，channels，rows，cols）的4D图像张量，则应设置规范化的轴为1
        #即设置axis=1 使得该网络层在规范化时沿着通道轴规范化
        #而我们使用的是tf后端要求的channel_last格式的4D图像张量，因此需要设置axis=-1
        '''
        keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                                                        gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                                        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
        axis: 整数，指定要规范化的轴，通常为特征轴。例如在进行data_format="channels_first的2D卷积后，一般会设axis=-1。
        momentum: 动态均值的动量
        epsilon：大于0的小浮点数，用于防止除0错误
        center: 若设为True，将会将beta作为偏置加上去，否则忽略参数beta
        scale: 若设为True，则会乘以gamma，否则不使用gamma。当下一层是线性的时，可以设False，因为scaling的操作将被下一层执行。
        beta_initializer：beta权重的初始方法
        gamma_initializer: gamma的初始化方法
        moving_mean_initializer: 动态均值的初始化方法
        moving_variance_initializer: 动态方差的初始化方法
        beta_regularizer: 可选的beta正则
        gamma_regularizer: 可选的gamma正则
        beta_constraint: 可选的beta约束
        gamma_constraint: 可选的gamma约束
        '''

        #Uncomment to use BN Layer during training
        #self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(128, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(128, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(256, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(256, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(512, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(512, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(512, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(512, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #11 池化层
        self.model.add(Dropout(0.25))                                       #12 Dropout层

        self.model.add(Flatten())                                           #13 Flatten层
        self.model.add(Dense(4096))                                          #14 Dense层,又被称作全连接层
        self.model.add(Activation('relu'))                                  #15 激活函数层
        self.model.add(Dropout(0.5))                                        #16 Dropout层

        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))                                   #17 Dense层
        self.model.add(Dropout(0.5))

        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))
        #输出模型概况
        self.model.summary()

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
    #test for building model
    VGG_16 = Model()
    VGG_16.build_model()