import os

from PIL import Image
import numpy as np
import tensorflow as tf
import argparse
import time
import vlc
import sys

try:
    from openvino import inference_engine as ie
    from openvino.inference_engine import IENetwork, IEPlugin
except Exception as e:
    exception_type = type(e).__name__
    print("The following error happened while importing Python API module:\n[ {} ] {}".format(exception_type, e))
    sys.exit(1)

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def pre_process_image(imagePath, img_height=299):
    # Model input format
    n, c, h, w = [1, 3, img_height, img_height]
    image = Image.open(imagePath)
    processedImg = image.resize((h, w), resample=Image.BILINEAR)

    # Normalize to keep data between 0 - 1
    processedImg = (np.array(processedImg) - 0) / 255.0

    # Change data layout from HWC to CHW
    processedImg = processedImg.transpose((2, 0, 1))
    processedImg = processedImg.reshape((n, c, h, w))

    return image, processedImg, imagePath

# Plugin initialization for specified device and load extensions library if specified.
parser = argparse.ArgumentParser()
parser.add_argument("--image", help="image to be processed")
args = parser.parse_args()
args_vars = vars(parser.parse_args())
plugin_dir = None
model_xml = '/opt/intel/openvino/deployment_tools/model_optimizer/optimized.xml'
model_bin = '/opt/intel/openvino/deployment_tools/model_optimizer/optimized.bin'
# Devices: GPU (intel), CPU, MYRIAD
plugin = IEPlugin("CPU", plugin_dirs=plugin_dir)
# Read IR
net = IENetwork.from_ir(model=model_xml, weights=model_bin)
assert len(net.inputs.keys()) == 1
assert len(net.outputs) == 1
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
# Load network to the plugin
exec_net = plugin.load(network=net)
del net

# Run inference
fileName = '/home/gason/demo_ltts/cash_recognition/test/a.jpg'
image, processedImg, imagePath = pre_process_image(args.image)

start = time.time()

res = exec_net.infer(inputs={input_blob: processedImg})

# Access the results and get the index of the highest confidence score
output_node_name = list(res.keys())[0]
res = res[output_node_name]
#print(res)
# Predicted class index.
idx = np.argsort(res[0])[-1]

#print(idx)

label_file = '/home/gason/demo_ltts/cash_recognition/models/labels.txt'
labels = load_labels(label_file)
#print(labels)
print(labels[idx], res[0][idx] * 100)
end = time.time()
print("[INFO] classification took " + str((end-start)*1000) + " ms")
time_taken = str((end-start)*1000)
write_time = open('/home/gason/demo_ltts/cash_recognition/demo_app/result_with_openvino.txt','w')
write_time.write(time_taken)
write_time.close()
