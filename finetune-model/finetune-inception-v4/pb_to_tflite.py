import tensorflow as tf
import os


save_path = './save_tflite'
if not os.path.exists(save_path):
    os.mkdir(save_path)


model_path = "../../../chenh/danger_driver/train_cnn_v1/model/test/saved_model.pb"

input_arrays = ["inputs_placeholder"]
output_arrays = ["predictions"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(model_path, input_arrays, output_arrays)
# quantize the model, which can minimize the model 1/4
converter.post_training_quantize = True

tflite_model = converter.convert()
open("converted_model_quantize.tflite", "wb").write(tflite_model)