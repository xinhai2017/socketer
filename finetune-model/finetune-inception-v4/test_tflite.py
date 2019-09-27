# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

test_image_dir = '../../../chenh_data/raw_data/split_train_test_face/test1/'
model_path = "./converted_model_quantize.tflite"

output_path = "./output_file"
if not os.path.exists(output_path):
    os.mkdir(output_path)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()

# print(str(input_details))
output_details = interpreter.get_output_details()
# print(str(output_details))

def parse_output(output_data):
    index = np.where(output_data == np.max(output_data))
    max_index = index[1][0]

    if int(max_index) == 0:
        predict_label = 'phone'
    elif int(max_index) == 1:
        predict_label = 'other'
    elif int(max_index) == 2:
        predict_label = 'drink'
    elif int(max_index) == 3:
        predict_label = 'smoke'
    return predict_label

def save_to_file(filename,test_image, predict):
    with open(filename,'a+') as f:
        f.write(test_image + ' ,' + predict + '\n')


def test():
    images = os.listdir(test_image_dir)
    for image in images:
        image_path = os.path.join(test_image_dir,image)
        img = cv2.imread(image_path)
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
        if predict in test_image_dir:
            save_to_file('./output_file/error.csv', test_image, predict)
        else:
            save_to_file('./output_file/output_file.csv', test_image, predict)


if __name__ == '__main__':
    test()