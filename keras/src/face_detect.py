# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
Date: 2018/11/21
Author: Xu Yucheng 1611453
Abstract: Code for face recognition
'''

import os
import sys
import cv2
import gc
from utils import *
import numpy as np
import random
from train_model import *
from utils import print_matrix
import PIL.Image as Image

cascade_path = '../../dataset/haarcascade_frontalface_alt.xml'
TEST_DATA = '../../dataset/TestData'
MODEL_PATH = '../model/KFold_Manual_model.h5'
IMAGE_SIZE = 64

images=[]
labels=[]

#def printUsage():


def find_largest_threshold(res):
    max = 0
    index = 0
    length = len(res)
    for i in range(length):
        if res[i] > max:
            max = res[i]
            index = i
    return index
    
def generate_orderSheet(path):
    #用来生成一个显示训练标签和实际标签之间映射关系的数组
    folders = os.listdir(path)
    folders = np.asarray(folders, dtype=np.float32)
    order_sheet = np.array([], dtype=np.uint32)
    mini_id = min(folders)
    for folder in folders:
        temp = int(folder) - mini_id
        order_sheet = np.append(order_sheet, temp)
    order_sheet = sorted(order_sheet)
    return order_sheet, mini_id

def generate_testdata(path):
    order_sheet, mini_id = generate_orderSheet(path)
    #从每一个类别中随机抽取一张照片输入网络进行测试
    folders = os.listdir(path)
    for folder in folders:
        full_path = os.path.abspath (os.path.join(path,folder))
        index = random.randint(10000, 10047)
        filename = str(index) + '.png'
        full_path = os.path.abspath(os.path.join(full_path, filename))
        image = cv2.cvtColor (
                        resize_image(cv2.imread(full_path), padding=True), cv2.COLOR_BGR2RGB
                    ).astype(np.float32)/255
        global images
        images.append(image)

        for i in range(len(folders)):
            temp = int(folder) - mini_id
            if temp == order_sheet[i]:
                label = i
        global labels
        labels.append(label)
    #转化为网络需要的np array格式
    images = np.array(images)
    labels = np.array(labels)  

def cal_accuracy(predicts, labels):
    total = len(labels)
    correct = 0
    for i in range (total):
        if predicts[i] == labels[i]:
            correct +=1
    return correct/total

def image_visualization():
    SIZE = images.shape[1]
    to_image = Image.new('RGB', (SIZE*8, SIZE*8))
    for row in range(8):
        if row == 7:
            for col in range(6):
                from_image = Image.fromarray((images[row*8+col]*255).astype('uint8')).convert('RGB')
                to_image.paste(from_image, (col*SIZE, row*SIZE))    
        else:
            for col in range(8):
                from_image = Image.fromarray((images[row*8+col]*255).astype('uint8')).convert('RGB')
                to_image.paste(from_image, (col*SIZE, row*SIZE))
    return to_image.save('../test.png')
        

def face_detect_main(options, cam_id=0):
    #装载训练好的模型
    model = Model()
    model.load_model(MODEL_PATH)
    #识别部分，分为使用图片进行识别，和利用摄像头中的实时图像进行识别
    if options == 'use_images':
        #如果选择使用现有图片来预测，则从62类的照片中每类任意抽出一张进行预测
        order_sheet, mini_id = generate_orderSheet(TEST_DATA)
        generate_testdata(TEST_DATA)
        image_visualization()
        predict = model.predict(images,image_num=len(labels))
        for i in range (len(labels)):
            predict[i] = order_sheet[predict[i]] + mini_id 
            labels[i] = order_sheet[labels[i]] + mini_id
        print ('[predict]')
        print_matrix(predict)
        print ('[truth]')
        print_matrix(labels)
        print ('accuracy: %d'%(cal_accuracy(predict, labels)*100)+'%') 
        image = Image.open('../test.png')
        image.show()
        
    
    elif options == 'use_camera':
        #默认使用电脑自带摄像头进行预测，修改camera_id可以选择使用外接摄像头
        camera_id = cam_id
        color = (255,0,255)
        cap = cv2.VideoCapture(camera_id)
        order_sheet, mini_id = generate_orderSheet(TEST_DATA)
        while True:
            #获取摄像头中实时图片并灰度化，减少分割出人脸时的计算量
            _, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #加载cascade分类器,并从图像中分割出人脸部分
            cascade = cv2.CascadeClassifier(cascade_path)
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(48,48), maxSize=(192,192))

            if len(faceRects)>0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect
                    #send face image to network
                    image = frame[y - 30: y + h + 30, x - 30: x + w + 30]
                    image = cv2.cvtColor(
                        resize_image(image, padding=True), cv2.COLOR_BGR2RGB
                    ).astype(np.float32)/255
                    image = image.reshape(1,64,64,3)
                    faceID = model.predict(image)
                    #using faceID to set the annotation
                    label = str(order_sheet[faceID] + mini_id)
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

                    #文字提示是谁
                    cv2.putText(frame,label,
                                (x + 10, y - 40),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                color,                                 #颜色
                                2)                                     #字的线宽
            cv2.imshow("face_detect", frame)

            #等待10毫秒看是否有按键输入
            k = cv2.waitKey(10)
            #如果输入q则退出循环
            if k & 0xFF == ord('q'):
                break
        #释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()
        

if __name__ == '__main__':
    face_detect_main('use_camera')