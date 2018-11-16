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

#load the predifined function to load all the images to dataset
from loadFaceDataset import load_dataset, IMAGE_SIZE, DataBase_Root_Path
#The ROOT path of DataSet

MODEL_PATH = '/home/kamerider/Machine Intelligent/face_recognition/model/face_recognition.h5'
HISTORY_PATH = '/home/kamerider/Machine Intelligent/face_recognition/History/Train_History.txt'
FIGURE_PATH = '/home/kamerider/Machine Intelligent/face_recognition/History'

class DataSet:
    def __init__(self, path_name):
        #train set
        self.train_images = None
        self.train_labels = None

        #valid set
        #self.valid_images = None
        #self.valid_labels = None

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

        print 'generate labels'
        #print labels
        #shuffle the DataSet
        print 'shuffle the dataset '
        index = np.arange(len(labels))
        np.random.shuffle(index)
        print (index[0:20])
        images = images[index,:,:,:]
        labels = labels[index]
        #train_images, valid_images, train_labels, valid_labels = train_test_split(images,labels,test_size = 0.2, random_state = 0)
        #train_images, valid_images, train_labels,valid_labels = train_test_split(images, labels, test_size = 0, random_state = random.randint(0,102))
        #_, valid_images, _,valid_labels = train_test_split(images, labels, test_size = 0.33, random_state =random.randint(0,102))

        #测试集是否需要与训练集无交集？
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.25, random_state =random.randint(0,102), shuffle = True)
        
        

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
        #if using Theano as backend make a reshape for every image
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], image_channels, image_rows, image_cols)
            #valid_images = valid_images.reshape(valid_images.shape[0], image_channels, image_rows, image_cols)
            test_images = test_images.reshape(test_images.shape[0], image_channels, image_rows, image_cols)
            self.input_shape = (image_channels, image_rows, image_cols)


        if K.image_dim_ordering() == 'tf':
            train_images = train_images.reshape(train_images.shape[0], image_rows, image_cols, image_channels)
            #valid_images = valid_images.reshape(valid_images.shape[0], image_rows, image_cols, image_channels)
            test_images = test_images.reshape(test_images.shape[0], image_rows, image_cols, image_channels)
            self.input_shape = (image_rows, image_cols, image_channels)

        print (train_images.shape[0], 'train samples')
        #print (valid_images.shape[0], 'valid samples')
        print (test_images.shape[0], 'test samples')

        #using One_hot codeing to deal with our labels and transform it to a vector
        #using categorical_crossentropy as loss-function
        train_labels =np_utils.to_categorical(train_labels, predifined_class)
        #valid_labels =np_utils.to_categorical(valid_labels, predifined_class)
        test_labels =np_utils.to_categorical(test_labels, predifined_class)

        #floatilize the pixel
        train_iamges = train_images.astype('float32')
        #valid_images = valid_images.astype('float32')
        test_images  = test_images.astype('float32')

        #image normalization
        train_images /= 255
        #valid_images /= 255
        test_images  /= 255

        #assign the self varibles
        self.train_images = train_iamges
        #self.valid_images = valid_images
        self.test_images  = test_images
        self.train_labels = train_labels
        #self.valid_labels = valid_labels
        self.test_labels  = test_labels


class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes = 5):
        #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        #使用了一个标准的VGG-16网络结构
        #输入之前需要将图像resize成为224x224的大小
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
        axis: 整数，指定要规范化的轴，通常为特征轴。例如在进行data_format="channels_first的2D卷积后，一般会设axis=1。
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
        self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
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

    #训练模型
    def train(self, dataset, batch_size = 32, nb_epoch = 10, data_augmentation = False):
        sgd = SGD(lr = 0.01, decay = 1e-6,
                  momentum = 0.9, nesterov = True) #采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
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
                           validation_split = 0.33,
                           #validation_data=(dataset.valid_images, dataset.valid_labels),
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
            epochs = np.arange(nb_epoch)+1
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
            datagen.fit(dataset.train_images)

            #利用生成器开始训练模型
            hist = self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                   batch_size = batch_size),
                                     samples_per_epoch = dataset.train_images.shape[0],
                                     nb_epoch = nb_epoch,
                                     validation_split = 0.33,
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
            epochs = np.arange(nb_epoch)+1
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
