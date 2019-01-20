#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2019/1/11
Author: Xu Yucheng
Abstract: Code for face detection
'''
import os
import torch 
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import matplotlib.pyplot as plt
from create_test_data import find_image, TEST_DATA

#训练完的模型存放位置
MODEL_PATH = '../models/net.pkl'

def load_net(path=MODEL_PATH):
    net = torch.load(path)
    return net

def imshow(image):
    image = image/2 + 0.5
    np_image = image.numpy()
    plt.imshow(np.transpose(np_image, (1,2,0)))
    plt.show()

#pytorch中就只写了简单地抽取样本进行测试
def test_model():
    #先生成测试用数据， 每次从全部的测试集中每一类抽取一张
    find_image()
    net = load_net()
    test_set = torchvision.datasets.ImageFolder(
        TEST_DATA,transform=transforms.Compose([
                                    transforms.Resize((64,64)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5229, 0.4207, 0.3647), (0.2120, 0.1877, 0.1734)),])
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=62, 
        shuffle=True, num_workers=4
        )
    data_iter = iter(test_loader)
    images, labels = data_iter.next()
    output = net(Variable(images.cuda()))
    _, predicted = torch.max(output.data, 1)
    print ('[Truth]: ')
    print (labels.data)
    print ('[Predict]: ')
    print (predicted.data)
    imshow(torchvision.utils.make_grid(images,nrow=8))

if __name__ == '__main__':
    test_model()



