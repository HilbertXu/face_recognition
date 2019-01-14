#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2019/1/11
Author: Xu Yucheng
Abstract: Code for loading dataset through pytorch api
'''
import os
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms

TRAIN_DATASET = '/home/kamerider/Documents/Dataset_TF/dataset_train'
VALID_DATASET = '/home/kamerider/Documents/Dataset_TF/dataset_valid'

def load_train_data(path=TRAIN_DATASET):
    trainset = torchvision.datasets.ImageFolder(path, 
                                                transform=transforms.Compose([
                                                    transforms.Resize((64,64)),
                                                    transforms.Centercrop(64),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    #此处训练图像的均值和标准差由utils.py中函数算得
                                                    transforms.Normalize((0.5229, 0.4207, 0.3647), (0.2120, 0.1877, 0.1734)),]))
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=64, 
        shuffle=True, num_workers=4
        )
    return train_loader

def load_valid_data(path=VALID_DATASET):
    validset = torchvision.datasets.ImageFolder(path, 
                                                transform=transforms.Compose([
                                                    transforms.Resize((64,64)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
    valid_loader = torch.utils.data.DataLoader(
        trainset, batch_size=64, 
        shuffle=True, num_workers=4
        )
    return valid_loader

def get_class(path=TRAIN_DATASET):
    classes = os.listdir(TRAIN_DATASET)
    return classes

def generate_dataset(option):
    train_loader = load_train_data()
    valid_loader = load_valid_data()
    classes = get_class()
    if option == 'train':
        return train_loader, classes
    elif option == 'valid':
        return valid_loader
    