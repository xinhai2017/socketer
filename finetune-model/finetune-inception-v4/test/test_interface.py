# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
"""

import numpy as np
import os
import cv2
from predict import *

alg_core = TEAlg(pb_path_1="../model/frozen_model.pb")

File = open("output.csv", "w")
errorfile = open("error.csv", "w")

File.write('test_image: ' + '     , ' + 'predict_result:' + "\n")
errorfile.write('test_image: ' + '     , ' + 'predict_result:' + "\n")

def load_img(imgDir):
    image_names = os.listdir(imgDir)
    count = 0
    for image_name in image_names:
        image_path = os.path.join(imgDir,image_name)
        img = cv2.imread(image_path)
        if img is not None:
            result_dict = ProjectInterface({image_path: image_path}, proxy=alg_core)
            test_image, predict_result = parser_result(result_dict)
            if predict_result in test_image:
                count += 1
            else:
                errorfile.write(test_image + "     , " + predict_result + "\n")
            File.write(test_image + "     , " + predict_result + "\n")
    print('true prediction number/total images number: %d/%d' %(count,len(image_names)))

def parser_result(result):
    for key, value in result.items():
        key = key
        value = value
    test_image = key.split('/')[-1]
    # print("test image :", test_image)

    max_index = max(value, key=value.get)
    if int(max_index) == 0:
        predict_result = 'phone'
        # print('predict result: phone')

    elif int(max_index) == 1:
        predict_result = 'other'
        # print('predict result: other')

    elif int(max_index) == 2:
        predict_result = 'drink'
        # print('predict result: drink')

    elif int(max_index) == 3:
        predict_result = 'smoke'
        # print('predict result: smoke')

    return test_image, predict_result

def load_database(imgDir):
    data = load_img(imgDir)
    train_imgs = np.asarray(data)
    return train_imgs

def test():
    craterDir = "../../../../chenh_data/raw_data/split_train_test_face/test1/"
    load_database(craterDir)
    File.close()


if __name__ == '__main__':
    test()
