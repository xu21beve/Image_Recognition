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
ADD_ONS = [['mayo', 'trash', 'plastic', 13.2],
           ['mustard', 'trash', 'plastic', 6.1],
           ['chili', 'trash', 'plastic', 7.7],
           ['compostable-fork', 'compost', 'compostable-plastic', 5.25],
           # ['compostable-spoon', 'compost', 'compostable-plastic', -1],
           ['one-use-fork', 'trash', 'plastic', -1]]
           # ['one-use-knife', 'trash', 'plastic', '-1'],
           # ['napkin', 'compost, recycling', 'unwaxed-paper', '-1'],
           # ]

# ADD_ONS = [['compostable-fork', 'compost', 'compostable-plastic', '-1'],
#            ['compostable-spoon', 'compost', 'compostable-plastic', '-1'],
#            ['one-use-fork', 'trash', 'plastic', '-1'],
#            ['one-use-knife', 'trash', 'plastic', '-1'],
#            ['napkin', 'compost, recycling', 'unwaxed-paper', '-1'],
#            ['condiments-packet', 'trash', 'plastic', '-1']]

significant_weight_deviation = 20 # in grams
identification_threshold = 0.3 # probability

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def identify_utensils(image_path, data):
  image_file = image_path
  model_file = '../models/utensils_model.tflite'
  label_file = '../models/utensils_labels.txt'
  input_mean = 127.5
  input_std = 127.5
  num_threads = None
  
  ext_delegate = None
  ext_delegate_options = {}

  interpreter = Interpreter(
      model_path=model_file,
      experimental_delegates=ext_delegate,
      num_threads=None)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(image_file).resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)
  object_ids = []
  highest_probability_id = -1

  for i in results:
    data += i + ", "

  for i in top_k:
    if floating_model:
      print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
      if float(results[i]) > identification_threshold:  # 0.8 is the threshold for identification
        object_ids.append(i)
        if results[i] > results[highest_probability_id]:    # Choose item with the highest probability amongst the items with the five highest probabilities
            highest_probability_id = i 
    else:
      print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
  
  return object_ids
