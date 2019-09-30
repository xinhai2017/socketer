# -*- coding:utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
import tensorflow as tf
from multiprocessing.pool import Pool

test_image_dir = './test1/'
model_path = "./converted_model.tflite"

output_path = "./output_files_no_quantize_model"
if not os.path.exists(output_path):
    os.mkdir(output_path)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
# print(str(input_details)) # cat input arrays name.

output_details = interpreter.get_output_details()
# print(str(output_details)) # cat output arrays name.

def parse_output(output_data):
    index = np.where(output_data == np.max(output_data))
    max_index = index[1][0]

    if int(max_index) == 0:
        predict_label = 'phone'
        # print('predict result: phone')
    elif int(max_index) == 1:
        predict_label = 'other'
        # print('predict result: other')
    elif int(max_index) == 2:
        predict_label = 'drink'
        # print('predict result: drink')
    elif int(max_index) == 3:
        predict_label = 'smoke'
        # print('predict result: smoke')
    return predict_label

def save_to_file(filename,test_image, predict):
    with open(filename,'a+') as f:
        f.write(test_image + ' ,' + predict + '\n')


def test_tflite():
    images = os.listdir(test_image_dir)
    for image in images:
        image_path = os.path.join(test_image_dir,image)
        img = cv2.imread(image_path)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        input_data = cv2.resize(img, (299, 299))
        input_data = np.array(input_data, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data,axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predict = parse_output(output_data)
        test_image = image.split('/')[-1]
        if predict in test_image:
            # save_to_file('%s/output_file.csv' %output_path , test_image, predict)
            pass
        else:
            save_to_file('%s/error.csv' %output_path, test_image, predict)


if __name__ == '__main__':
    pool = Pool(8)

    result1 = pool.apply_async(test_tflite, )

    pool.close()
    pool.join()

    # test_tflite()