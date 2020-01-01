def display():
  # usage : python classify.py --image /home/user/Desktop/varun/cash_recognition/test/b.jpg
  import argparse

  import numpy as np
  import tensorflow as tf
  import cv2
  import streamlit as st
  import vlc
  import os
  import time

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


  if __name__ == "__main__":

    

    file_name = "/home/user/Desktop/varun/cash_recognition/test/b.jpg"
    model_file ="/home/user/Desktop/varun/cash_recognition/models/optimized.pb"
    label_file = "/home/user/Desktop/varun/cash_recognition/models/labels.txt"
    audio_file = "/home/user/Desktop/varun/cash_recognition/sounds/"
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

    with tf.compat.v1.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
          input_operation.outputs[0]: t
      })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    print(labels)
    print(top_k)
    print(type(results))
    for i in top_k:
      print(labels[i], results[i])

    image_copy = cv2.imread(args_vars["image"])
    idxs = np.argsort(results)[::-1][:1]

    for (i, j) in enumerate(idxs):
      # build the label and draw the label on the image
      label = "{}: {:.2f}%".format(labels[j], results[j] * 100)
      cv2.putText(image_copy, label, (100,100*2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 15)
      cv2.imwrite('output.jpg',image_copy)

      print(audio_file+str(labels[j])+'.wav')

      
      p = vlc.MediaPlayer(audio_file+str(labels[j])+'.wav')  
      p.play()
      time.sleep(2)
      
    
    resize_image = cv2.resize(image_copy,(480,640),interpolation = cv2.INTER_LINEAR)
    
    #cv2.imshow("Output", resize_image)
    #cv2.waitKey(0)

