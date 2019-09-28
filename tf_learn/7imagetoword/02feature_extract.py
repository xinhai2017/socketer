import os
import sys
import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging
import pprint
import pickle
import numpy as np

model_file = './datas/checkpoint_inception_v3/inception_v3_graph_def.pb'
input_description_file = './datas/results_20130124.token'
input_img_dir = './datas/flickr30k-images'
output_folder = './datas/download_inception_v3_features'

batch_size = 1000
if not gfile.Exists(output_folder):
    gfile.MakeDirs(output_folder)

def parse_token_file(token_file):
    """Parses image description file."""
    img_name_to_tokens = { }
    with gfile.GFile(token_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        img_id, description = line.strip('\r\n').split('\t')
        img_name, _ = img_id.split('#')
        img_name_to_tokens.setdefault(img_name, [])
        img_name_to_tokens[img_name].append(description)
    return img_name_to_tokens

img_name_to_tokens = parse_token_file(input_description_file)
all_img_names = list(img_name_to_tokens.keys())

# pprint.pprint("num of all images: %d" % len(all_img_names))
# pprint.pprint(all_img_names[0:10])
# pprint.pprint(img_name_to_tokens['2778832101.jpg'])

def load_pretrained_inception_v3(model_file):
    with gfile.FastGFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="")
load_pretrained_inception_v3(model_file)

num_batches = int(len(all_img_names) / batch_size)
if len(all_img_names) % batch_size != 0:
    num_batches += 1

with tf.Session() as sess:
    second_to_last_tensor = sess.graph.get_tensor_by_name("pool_3:0")
    for i in range(num_batches):
        batch_img_names = all_img_names[i*batch_size: (i+1)*batch_size]
        batch_features = []
        for img_name in batch_img_names:
            img_path = os.path.join(input_img_dir, img_name)
            pprint.pprint("processing img %s" % img_name)
            if not gfile.Exists(img_path):
                continue
            img_data = gfile.FastGFile(img_path,'rb').read()
            feature_vector = sess.run(second_to_last_tensor,
                                      feed_dict={
                                          "DecodeJpeg/contents:0": img_data
                                      })
            batch_features.append(feature_vector)
        batch_features = np.vstack(batch_features)
        output_filename = os.path.join(output_folder,"image_features_%d.pickle" % i)
        logging.info("writing to file %s" % output_filename)
        with gfile.GFile(output_filename,'w') as f:
            pickle.dump((batch_img_names, batch_features),f)