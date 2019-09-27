# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import time

import tensorflow as tf


model_path = "./converted_model.tflite"

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(str(input_details))
output_details = interpreter.get_output_details()
print(str(output_details))
