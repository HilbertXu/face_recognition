# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import sys
import cv2
import gc
import numpy as np
from trainModelWithKeras import Model, MODEL_PATH

cascade_path = '/home/kamerider/catkin_ws/src/machine_vision/haarcascade_frontalface_alt.xml'
DataBase_ROOT = '/home/kamerider/Machine Intelligent/face_recognition/DataBase'

#change this path to read different test image
IMAGE_PATH = '/home/kamerider/catkin_ws/src/machine_vision/body_image/1611472/body'
IMAGE_SIZE = 224

def findLargestThreshold(res):
    max = 0
    index = 0
    length = len(res)
    for i in range(length):
        if res[i] > max:
            max = res[i]
            index = i
    return index

def BubbleSort(array):
    order = array.copy()
    order = sorted(array)

    return order

def findMininumLabel(labels):
    min_label = 9999999
    for label in labels:
        int_label = int(label)
        if int_label < min_label:
            min_label = int_label
    return min_label

def generateOrderSheet(path):
    #Firstly, generate the class_sheet
    folders = os.listdir(DataBase_ROOT)
    print 'current classes is '
    print folders
    class_num = len(folders)
    min_index = findMininumLabel(folders)

    #create empty np array to contain the current int label & order sheet
    int_label = np.zeros(shape=(class_num,1))
    order_array = np.zeros(shape=(class_num,1))

    #Secondly convert string labels to int labels(standardlized)
    index = 0
    for folder in folders:
        temp = int(folder) - min_index
        int_label[index] = int(temp)
        index += 1

    #Thirdly generate the order_sheet
    order_array = BubbleSort(int_label)

    return order_array,min_index


#Resize face part to 128X128
#in order to suit the input size of CNN
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    #get the size of image
    h, w, _ = image.shape

    #if width != height
    #get the longer one
    longest_edge = max(h, w)

    #calculate how much pixel should be add to the shorter side
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    #RGB
    #set the border color
    BLACK = [0, 0, 0]

    #border
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)

    #resize image & return
    return cv2.resize(constant, (height, width))

if __name__ == '__main__':

    '''
    UNCOMMENT FOR TEST
    print len(sys.argv)
    for i in range(0,len(sys.argv)):
        print sys.argv[i]
    '''


    if len(sys.argv) != 3:
        print ("Usage: python facePredictWithKeras.py -[Options] <camera_id / image_path>")
        print ("Option: -cam        Using camera (default camera_id is 0)")
        print ("Option: -img        Using image")
        exit(0)

    #set the color of bounding boxes
    color = (255,0,255)

    model = Model()
    model.load_model(MODEL_PATH)

    #generate order_sheet
    order_sheet, standar_index = generateOrderSheet(DataBase_ROOT)
    print 'current order sheet is '
    print order_sheet

    

    if sys.argv[1] == "-cam":
        #capture image from real-time video or video files
        cap = cv2.VideoCapture(int(sys.argv[2]))
        while True:
            #read one image from video
            _, frame = cap.read()

            #graying image
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #loading classifier
            cascade = cv2.CascadeClassifier(cascade_path)

            #detect face part and resize
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect

                    #send face image to network
                    image = frame[y - 30: y + h + 30, x - 30: x + w + 30]
                    image = resize_image(image)
                    print image.shape
                    faceID = model.face_predict(image)
                    print faceID
                    #using faceID to set the annotation
                    label = str(order_sheet[faceID] + standar_index)

                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

                    #文字提示是谁
                    cv2.putText(frame,label,
                                (x + 10, y - 40),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
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
    
    if sys.argv[1] == "-img":
        while True:
            #read image from path
            path = sys.argv[2]
            frame = cv2.imread(sys.argv[2])
            frame = resize_image(frame)

            faceID = model.face_predict(frame)
            label = str(order_sheet[faceID] + standar_index)

            print ("result is: " + label)
            cv2.imshow("face_detect", frame)
            #等待10毫秒看是否有按键输入
            k = cv2.waitKey(10)
            #如果输入q则退出循环
            if k & 0xFF == ord('q'):
                break
