# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
Date: 2018/11/21
Author: Xu Yucheng 1611453
Abstract: Code for the whole system(Pytorch)
'''
import os
import sys

if __name__ == '__main__':
    print ("[My NAME] 1611453 徐宇成")
    print ("[PROJECT] Face Recognition System based on Tensorflow(keras) and Pytorch")
    print ("[INFO] Please choose which frame you want to use ")
    frame = input (" (Keras / Pytorch) ")
    if frame == 'keras' or frame == 'Keras':
        print ("[INFO] Now switch to Keras directory")
        current_path = os.getcwd()
        frame_path = os.path.abspath(os.path.join(current_path, 'keras/src'))
        os.chdir(frame_path)
        os.system("python keras_main.py")

    if frame == 'Pytorch' or frame == 'pytorch':
        print ("[INFO] Now switch to Pytorch directory")
        current_path = os.getcwd()
        frame_path = os.path.abspath(os.path.join(current_path, 'pytorch/src'))
        os.chdir(frame_path)
        os.system("python pytorch_main.py")