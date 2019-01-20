#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2018/11/23
Author: Xu Yucheng
Abstract: Code for create test data
'''
import os
import sys
import cv2
import shutil
import random

DATA_BASE = '../../dataset/TestData'
TEST_DATA = '../TestData'
DES_DIR = '../TestData'


def copy_image(from_path, to_path):
    shutil.copy(from_path, to_path)

def resize_image(image):
    return cv2.resize_image(image,(64,64))

def get_id(str1, str2):
    length = len(str2)
    label = str1[length+1:]
    return label

def find_image(path=DATA_BASE):
    os.chdir(TEST_DATA)
    os.system("rm -rf *")
    labels = os.listdir(path)
    for label in labels:
        DES_DIR = os.path.abspath(os.path.join(TEST_DATA,label))
        if not os.path.exists(DES_DIR):
            os.chdir(TEST_DATA)
            os.mkdir(label)
        folder_path = os.path.abspath(os.path.join(DATA_BASE, label))
        #for _ in range(10) means create 10 random int
        # _ can be replaced by any char
        indexes = [random.randint(10000,10047)]
        for index in indexes:
            filename = str(index) + '.png'
            filepath = os.path.abspath(os.path.join(folder_path, filename))
            copy_image(filepath, DES_DIR)
        print("\033[0;31;40m Images in %s have been moved to %s\033[0m"%(folder_path, DES_DIR))
    return 0




            


