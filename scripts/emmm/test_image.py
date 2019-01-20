# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import sys
import cv2
import gc
import PIL.Image as Image
import numpy as np
import random

TEST_DATA = '/home/kamerider/machine_learning/face_recognition/dataset/TestData'
IMAGE_SIZE = 128

images=[]
labels=[]
def generate_testdata(path):
    #order_sheet, mini_id = generate_orderSheet(TEST_DATA)
    #从每一个类别中随机抽取一张照片输入网络进行测试
    folders = os.listdir(path)
    for folder in folders:
        full_path = os.path.abspath (os.path.join(path,folder))
        index = random.randint(10000, 10047)
        filename = str(index) + '.png'
        full_path = os.path.abspath(os.path.join(full_path, filename))
        image = cv2.cvtColor (
                        cv2.resize(cv2.imread(full_path),(128, 128)), cv2.COLOR_BGR2RGB
                    ).astype(np.float32)/255
        global images
        images.append(image)
        global labels
        labels.append(folder)

    #转化为网络需要的np array格式
    images = np.array(images)
    labels = np.array(labels)  
    print (images.shape)
    to_image = Image.new('RGB', (8*IMAGE_SIZE, 8*IMAGE_SIZE))

    for line in range(8):
        for row in range(8):
            image = Image.fromarray(images[i*])


if __name__ == '__main__':
    generate_testdata(TEST_DATA)

