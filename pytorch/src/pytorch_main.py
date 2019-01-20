# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
Date: 2018/11/21
Author: Xu Yucheng 1611453
Abstract: Code for the whole system(Pytorch)
'''
import os
import sys
from train import train_and_save
from detect import test_model

def main():
    print ("[INFO] Now Your are using Pytorch frame")
    print ("[WARNING] For now we can only use Images to test")
    print ("[INFO] Please choose what to do")
    option_1 = input(" (train / recognize) ")
    if option_1 == 'train':
        train_and_save()
    if option_1 == 'recognize':
        test_model() 

if __name__ == '__main__':
    main()
   