#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2019/1/11
Author: Xu Yucheng
Abstract: Code for train and save model
'''
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn 
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim  
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import numpy as np
from build_model import *
from load_data import *

#trainning parameters
BATCH_SIZE = 64
EPOCH_NUM = 100
LR = 0.01
USE_CUDA = torch.cuda.is_available()

def train_and_save():

    trainloader, classes = generate_dataset('train')
    validloader = generate_dataset('valid')
    net = VGG('VGG16')
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    if USE_CUDA:
        # move param and buffer to GPU
        net.cuda()
        # parallel use GPU
        if torch.cuda.device_count() == 1:
            net = torch.nn.DataParallel(net, device_ids=[0])
        else:
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()-1))
        # speed up slightly
        cudnn.benchmark = True

    for epoch in range(EPOCH_NUM):
        #定义变量方便记录loss
        running_loss=0
        running_acc=0
        total=0
        for batch_idx, data in enumerate(trainloader, 0):
            #从DataLoader中读取数据
            inputs, labels = data
            if USE_CUDA:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            #梯度置零，因为在方向传播的过程中会累加上一次循环的梯度
            optimizer.zero_grad()

            outputs = net(inputs) #将数据输入网络
            loss = criterion(outputs, labels) #计算loss
            loss.backward() #loss反向传播
            optimizer.step() #反向传播后更新参数
            running_loss += loss.item()

            #计算分类准确率
            prediction = torch.argmax(outputs, 1)
            #也可以用torch.eq()函数实现
            #running_acc += torch.eq(labels.data).cpu().sum()
            running_acc += (prediction == labels).sum().float()
            total += len(labels)
            running_acc = running_acc/total

            #每十次输出一次loss的平均值和acc的平均值
            if batch_idx%10 == 0:
                print ("[EPOCH %d/%d BATCH %d] loss: %.3f acc: %.3f"%(epoch+1, EPOCH_NUM, batch_idx+1, running_loss/10, running_acc))
                running_loss=0
        

        net.eval()
        eval_loss = 0
        eval_acc = 0
        total = 0
        for i, data in enumerate(validloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()

            prediction = torch.argmax(outputs, 1)
            eval_acc += (prediction == labels).sum().float()
            total += len(labels)
        print ("[EPOCH %d/%d] val_loss: %.3f val_acc: %.3f"%(epoch+1, EPOCH_NUM, eval_loss/total, eval_acc/total))

    print ("Finished Training")
    torch.save(net, '/home/kamerider/machine_learning/face_recognition/pytorch/models/net.pkl')
    torch.save(net, '/home/kamerider/machine_learning/face_recognition/pytorch/models/net_params.pkl')
            
            
            
if __name__ == '__main__':
    train_and_save()
    