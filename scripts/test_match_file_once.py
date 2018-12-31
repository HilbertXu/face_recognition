#!/usr/bin/env python
#-*- coding: utf-8 -*-
import tensorflow as tf

filename = '/home/kamerider/machine_learning/face_recognition/tensorflow/TFRecords/train.*'

train_file = tf.train.match_filenames_once(filename)

init = (tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(train_file))