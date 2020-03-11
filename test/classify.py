# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# usage : python classify.py --image /home/user/Desktop/varun/cash_recognition/test/b.jpg
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import cv2
import streamlit as st
import vlc
import os
import time
import matplotlib
import matplotlib.pyplot as plt

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.compat.v1.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def plot_preds(labels, preds):

  likeability_scores = np.array(preds)

  data_normalizer = matplotlib.colors.Normalize()
  color_map = matplotlib.colors.LinearSegmentedColormap(
    "my_map",
    {
        "red": [(0, 1.0, 1.0),
                (1.0, .5, .5)],
        "green": [(0, 0.5, 0.5),
                  (1.0, 0, 0)],
        "blue": [(0, 0.50, 0.5),
                 (1.0, 0, 0)]
    }
)
  
  y_pos = np.arange(len(labels))

  fig, ax = plt.subplots()
  ax.barh(y_pos, preds, color = color_map(data_normalizer(likeability_scores)),  edgecolor='black',align='center')
  ax.set_yticks(y_pos)
  plt.gcf().subplots_adjust(bottom=0.15,left=0.20)
  ax.set_yticklabels(labels)
  ax.invert_yaxis() 
  ax.set_xlabel('Probability')
  ax.set_ylabel('Classes')
  ax.set_title('Result')

  #plt.show()
  plt.savefig('plot.png')



if __name__ == "__main__":

  

  file_name = "/home/gason/demo_ltts/cash_recognition/test/v.jpg"
  model_file ="/home/gason/demo_ltts/cash_recognition/models/optimized.pb"
  label_file = "/home/gason/demo_ltts/cash_recognition/models/labels.txt"
  audio_file = "/home/gason/demo_ltts/cash_recognition/sounds/"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Mul"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()
  args_vars = vars(parser.parse_args())

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)
  
  
  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  start = time.time()
  with tf.compat.v1.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)
  #print(labels)
  #print(top_k)
  #print(type(results))
  '''
  for i in top_k:
    print(labels[i], results[i])
  '''

  image_copy = cv2.imread(args_vars["image"])
  idxs = np.argsort(results)[::-1][:1]

  end = time.time()
  for (i, j) in enumerate(idxs):
    # build the label and draw the label on the image
    label = "{}: {:.2f}%".format(labels[j], results[j] * 100)
    print("[INFO] classification took " + str((end-start)*1000) + " ms")
    time_taken = str((end-start)*1000)
    write_time = open('/home/gason/demo_ltts/cash_recognition/demo_app/result_without_openvino.txt','w')
    write_time.write(time_taken)
    write_time.close()

    cv2.putText(image_copy, label, (100,100*2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 15)
    cv2.imwrite('output.jpg',image_copy)

    print(audio_file+str(labels[j])+'.wav')

    
    p = vlc.MediaPlayer(audio_file+str(labels[j])+'.wav')  
    p.play()
    time.sleep(2)

    plot_preds(labels,results)
    
  
  #resize_image = cv2.resize(image_copy,(480,640),interpolation = cv2.INTER_LINEAR)
  
  #cv2.imshow("Output", resize_image)
  #cv2.waitKey(0)
