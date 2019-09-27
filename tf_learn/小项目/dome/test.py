# encoding: utf-8
'''
@author: 真梦行路
@file: test.py
@time: 18-11-20 下午9:28
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

v1 = tf.Variable(tf.constant([1, 0, 1, 0, 1,
                  0, -4, -4, -4, 0,
                  1, -4, 24, -4, 1,
                  0, -4, -4, -4, 0,
                  1, 0, 1, 0, 1],shape=[5,5,3,35],name='v1'))
# v1=tf.concat([v1,v1,v1],axis=0)
# v1=tf.reshape(v1,shape=[5,5,3])

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(v1))