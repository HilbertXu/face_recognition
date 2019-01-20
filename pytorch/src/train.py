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
from utils import *
from tensorboardX import SummaryWriter

#trainning parameters
BATCH_SIZE = 64
EPOCH_NUM = 10
LR = 0.01
USE_CUDA = torch.cuda.is_available()
MODEL_DIR = '../model'

def train_and_save():

    trainloader, classes = generate_dataset('train')
    validloader = generate_dataset('valid')
    net = VGG('VGG16')
    print (net)
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
    writer = SummaryWriter(log_dir='../log', comment='VGG16')
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

            #每十次输出一次loss的平均值和acc的平均值
            if batch_idx%10 == 0:
                correct=0
                #计算分类准确率
                prediction = torch.argmax(outputs, 1)
                #也可以用torch.eq()函数实现
                #correct += predicted.eq(targets.data).cpu().sum()
                correct += (prediction == labels).sum().float()
                running_acc = correct/len(labels)
                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            %((running_loss/10), 100.*running_acc, correct, len(labels)))
                running_loss=0
                running_acc=0
                total=0
        

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
    global MODEL_DIR
    print ("model has been saved to %s"%(os.path.abspath(MODEL_DIR)))
    torch.save(net, os.path.abspath(MODEL_DIR) + '/net.pkl')
    torch.save(net.state_dict(), os.path.abspath(MODEL_DIR) + '/net_params.pkl')

def net_test():
    # 先切到测试模型
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_loader = generate_dataset('test')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # loss is variable , if add it(+=loss) directly, there will be a bigger ang bigger graph.
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            
if __name__ == '__main__':
    train_and_save()
    