# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""label_image for tflite."""

import argparse
import time
import hx711

import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

            # name              bin         material             weight
ADD_ONS = [['compostable-fork', 'compost', 'compostable-plastic', '-1'],
           ['one-use-fork', 'trash', 'plastic', '-1'],
           ['one-use-knife', 'trash', 'plastic', '-1'],
           ['mayo', 'trash', 'plastic', '-1'],
           ['chili', 'trash', 'plastic', '-1']
           ['mustard', 'trash', 'plastic', '-1']]
# ADD_ONS = [['compostable-fork', 'compost', 'compostable-plastic', '-1'],
#            ['compostable-spoon', 'compost', 'compostable-plastic', '-1'],
#            ['one-use-fork', 'trash', 'plastic', '-1'],
#            ['one-use-knife', 'trash', 'plastic', '-1'],
#            ['napkin', 'compost, recycling', 'unwaxed-paper', '-1'],
#            ['condiments-packet', 'trash', 'plastic', '-1']]

significant_weight_deviation = 20 # in grams
identification_threshold = 0.8 # probability

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='../06:31:44.jpg',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='../model_unquant.tflite',
      help='.tflite model to be executed')
  parser.add_argument(
      '-l',
      '--label_file',
      default='../labels.txt',
      help='name of file containing labels')
  parser.add_argument(
      '--input_mean',
      default=127.5, type=float,
      help='input_mean')
  parser.add_argument(
      '--input_std',
      default=127.5, type=float,
      help='input standard deviation')
  parser.add_argument(
      '--num_threads', default=None, type=int, help='number of threads')
  parser.add_argument(
      '-e', '--ext_delegate', help='external_delegate_library path')
  parser.add_argument(
      '-o',
      '--ext_delegate_options',
      help='external delegate options, \
            format: "option1: value1; option2: value2"')

  args = parser.parse_args()

  ext_delegate = None
  ext_delegate_options = {}

  # parse extenal delegate options
  if args.ext_delegate_options is not None:
    options = args.ext_delegate_options.split(';')
    for o in options:
      kv = o.split(':')
      if (len(kv) == 2):
        ext_delegate_options[kv[0].strip()] = kv[1].strip()
      else:
        raise RuntimeError('Error parsing delegate option: ' + o)

  # load external delegate
  if args.ext_delegate is not None:
    print('Loading external delegate from {} with args: {}'.format(
        args.ext_delegate, ext_delegate_options))
    ext_delegate = [
        tflite.load_delegate(args.ext_delegate, ext_delegate_options)
    ]

  interpreter = Interpreter(
      model_path=args.model_file,
      experimental_delegates=ext_delegate,
      num_threads=args.num_threads)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(args.image).resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - args.input_mean) / args.input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(args.label_file)
  object_ids = []
  highest_probability_id = -1

  for i in top_k:
    if floating_model:
      print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
      if float(results[i]) > identification_threshold:  # 0.8 is the threshold for identification
        object_ids.append(i)
        if results[i] > results[highest_probability_id]:    # Choose item with the highest probability amongst the items with the five highest probabilities
            highest_probability_id = i 
    else:
      print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))