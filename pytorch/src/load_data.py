#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2019/1/11
Author: Xu Yucheng
Abstract: Code for loading dataset through pytorch api
'''
import torch
import torchvision
import torchvision.transforms as transforms

TRAIN_DATASET = '/home/kamerider/Documents/Dataset_TF/dataset_train'
VALID_DATASET = '/home/kamerider/Documents/Dataset_TF/dataset_valid'

def load_train_data(path=TRAIN_DATASET):
    trainset = torchvision.datasets.ImageFolder(path, 
                                                transform=transforms.Compose([
                                                    transforms.Resize((64,64)),
                                                    transforms.Centercrop(64),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
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
