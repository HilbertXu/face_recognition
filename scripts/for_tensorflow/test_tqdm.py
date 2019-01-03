#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2019/1/1
Author: Xu Yucheng
Abstract: Code for testing tqdm
'''
from tqdm import tqdm
from time import sleep

for i in tqdm(range(10)):
    print ("[TRANING...]")
    sleep(0.1)