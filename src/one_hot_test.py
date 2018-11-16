# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import sys
import numpy as np
import cv2
from keras.utils import np_utils

def BubbleSort(array):
    order = array
    print array
    nums = len(array)
    for i in range(nums - 1):
        for j in range(nums - i -1):
            if order[j] < order[j+1]:
                order[j], order[j+1] = order[j+1], order[j]

    return order

def findMin(labels):
    min = 9999999
    for label in labels:
        int_label = int(label)
        if int_label < min:
            min = int_label
            print 'current mininum label is ' + str(min)
    return min

def standard(labels):
    temp_labels = []
    standard = findMin(labels)
    np_array = np.zeros(shape=(5,1))
    order_array = np_array
    standard_array = np_array
    count = 0
    for label in labels:
        temp = int(label) - standard
        print temp
        np_array[count] = int(temp)
        print np_array[count]
    print np_array
    order_array = BubbleSort(np_array)
    for i in range(5-1):
        for j in range(5-1):
            if np_array[i] == order_array[j]:
                standard_array[i] = j
    return standard_array



if __name__ == '__main__':
    labels = ['1611453','1611472', '1611462', '1611437', '1611427']
    labels = np.asarray(labels, dtype = np.float32)
    labels = standard(labels)
    print labels


    '''
    order = np_array
    print np_array
    print order
    array = np_utils.to_categorical(np_array, 20)
    print array

    test = np.array([[0] ,[1], [2], [3]])
    test = np_utils.to_categorical(test, 5)
    print labels
    print 'this is the test np_array'
    print test
    '''
