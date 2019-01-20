# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
Date: 2018/11/21
Author: Xu Yucheng 1611453
Abstract: Code for the whole system
'''
import os
import sys
from face_detect import face_detect_main
from train_model import train_model_main

def print_usage(program_name):
    print ("=================================================")
    print ("\033[1;31;40m        [ERROR] Parameters Mismatch !         \033[0m")
    print ("=================================================\n")
    print (" \033[1;31;40mUsage of %s: \033[0m"%(program_name))
    print (" \033[1;31;40mPlease follow the guide in terminal \033[0m")
    print (" python %s "%(program_name))
    print (" options:")
    print ("-train                         strat training a new model")
    print ("-DSS(DataShuffleSplit)         Split train data & valid data manually")
    print ("-KFoldM                        Use KFold to split data, and use each fold of data to implement Cross Validation")
    print ("-KFoldW                        Use Scikit-learn wrapper for keras to implement Cross Validation")
    print ("-GridSearch                    Use Graid Search from Scikit-learn to tune hyper-parameter(optimizer, init_mode, nb_epoch, batch_size)")

    print ("-recognize                     Use a pre-trained model to recognize faces")
    print ("-use_image                     Use sample images to test model")
    print ("-use_camera                    Use camera to get real-time video and recognize faces in the video frame") 
    print ("-help                          Print This Help")
    print ("\033[1;31;40m PLZ DO NOT use %s for now, it may cause Memory Overflow\033[0m"%('-GridSearch'))

    exit(0)

if __name__ == '__main__':
    print ("徐宇成 1611453 face recognition system")
    print ("Please choose what to do")
    option_1 = input("(train / recognize ?) ")
    if option_1 == 'train':
        print ("Please choose train mode")
        option_2 = input("(DataShuffleSplit(DSS) / KFoldCrossValidation(KFD) ?) ")
        if option_2 == "KFD":
            print ("There are two ways for KFold: Manual and Wrapper")
            print ("Please choose one")
            option_3 = input("(KFoldM / KFoldW ?) ")
            options = [option_1, option_2, option_3]
            print ('options: ' + str(options))
            train_model_main(option_3)


        elif option_2 == 'DSS':
            options = [option_1, option_2]
            print ('options: ' + str(options))
            train_model_main(option_2)
        
        else:
            print_usage(sys.argv[0])
        
    elif option_1 == 'recognize':
        print ("Please choose recognize mode")
        option_2 = input ("(use_images / use_camera ?) ")
        options = [option_1, option_2]
        print ('options: ' + str(options))
        face_detect_main(option_2)
    
    else:
        print_usage(sys.argv[0])


    