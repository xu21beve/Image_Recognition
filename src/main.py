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
import sys

import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter
from weight import totalWeight
from add_ons import identify_utensils


# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(3, 3), dtype=np.uint8)  # len(classes) --> substitute for size because classes not initialized

              # name              bin       material   weight
CONTAINERS = [['compostable-cup', 'compost', 'compostable-plastic', 8.6],
              ['condiments-cup', 'recycle', 'plastic', 4],
              ['small-waxed-paper-tray', 'trash', 'waxed-paper', 6],
              ['soup-container', 'recycle', 'plastic', 28.65],
              ['takeaway-container', 'recycling', 'plastic', 14.7],
              ['tin-no-lid', 'recycling', 'metal', 7.15],
              ['tin-with-lid', 'recycling', 'metal, aluminum', 11.5], # Can instruct to just recycle away lid
              ['large-waxed-paper-tray', 'trash', 'waxed-paper', 14.2],
              ['large-unwaxed-paper-tray', 'recycling', 'unwaxed-paper', 10.65]]

            # name              bin         material             weight
ADD_ONS = [# ['mayo', 'trash', 'plastic', 13.2],
           ['mustard', 'trash', 'plastic', 6.1],
           ['chili', 'trash', 'plastic', 7.7],
           ['compostable-fork', 'compost', 'compostable-plastic', 5.25],
           # ['compostable-spoon', 'compost', 'compostable-plastic', -1],
           ['one-use-fork', 'trash', 'plastic', -1]]
           # ['one-use-knife', 'trash', 'plastic', '-1'],
           # ['napkin', 'compost, recycling', 'unwaxed-paper', '-1'],
           # ]

significant_weight_deviation = 20 # in grams
identification_threshold = 0.5 # probability

# weigh = hx711(5, 6) # change to actual pins

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='../image.jpg',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='../models/containers_model.tflite',
      help='.tflite model to be executed')
  parser.add_argument(
      '-l',
      '--label_file',
      default='../models/containers_labels.txt',
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
  
  for i in range(20):
    print(f"[wait time]: {i*5}%")
    time.sleep(0.2)

  for i in top_k:
    if floating_model:
      print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
      if float(results[i]) > identification_threshold:  # 0.8 is the threshold for identification
        object_ids.append(i)
        if results[i] > results[highest_probability_id]:    # Choose item with the highest probability amongst the items with the five highest probabilities
            highest_probability_id = i 
    else:
      print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

  for i in object_ids:
    print(f"Object Identified: {labels[i]}")    # Initial testing for bin TODO: Add logic following probability table
    print(f"Bin: {CONTAINERS[i][1]}")
  if len(object_ids) < 1:
    print(f"No object identified with greater than {identification_threshold} probability")
    
  wrappers_cutlery = identify_utensils(args.image)
  weight = CONTAINERS[highest_probability_id][3]  # total weight
  for i in wrappers_cutlery:
    weight += ADD_ONS[i][3]
  print(f"expected weight: {weight}")

  # Start processing for most likely container
  if CONTAINERS[highest_probability_id][1] == "compost":
    # function to look for wrappers/cutlery
        # if identified, instruct dispoal of wrappers/cutlery, then
    for i in wrappers_cutlery:
      if ADD_ONS[i][1] != 'recycling':
        print(f"Throw {ADD_ONS[i][0]} into {ADD_ONS[i][1]}")
    # return "compost"
    print("Compost")
  else:
    # Look for wrappers/cutlery -- don't have access to these right now, so for now, just add an informational page about the different utensils
    # if identified, instruct dispoal of wrappers/cutlery, then
    condiments = False
    for i in wrappers_cutlery:
      if (ADD_ONS[i][1] != CONTAINERS[highest_probability_id][1] or not (highest_probability_id == -1 and ADD_ONS[i][1] == "trash")) and not condiments:
        if i == 0 or i == 1:
          condiments = True
          print(f"Throw condiment packets into {ADD_ONS[i][1]}")
        else:
          print(f"Throw {ADD_ONS[i][0]} into {ADD_ONS[i][1]}")
        
    if highest_probability_id == -1:
      print("Throw the rest in the trash")
      sys.exit(0)
      
    if totalWeight() - weight > significant_weight_deviation: # calculate using oil? bc less dense than water
    # If weight deviation (including any wrappers/cutlery) is significant there is food
      if CONTAINERS[highest_probability_id][2] == "soup-container":
        print("Dump remaining contents (food, napkins, and compostable utensils) into compost. Then recycle container")
      # if not soup container
      elif CONTAINERS[highest_probability_id][2] == "unwaxed-paper":
        print("Compost")
      else:
        if CONTAINERS[highest_probability_id][0] == "tin-with-lid":
          print("Recycle plastic lid")
        print(f"Dump the food into compost, and then throw {CONTAINERS[highest_probability_id][0]} in trash.")
    elif CONTAINERS[highest_probability_id][1] == "recycling": # if weight deviation (including any wrappers/cutlery) is not significant there is no food
        print("Recycle")
    else:
      print("Throw all remaining into trash.")
            
print(f"{totalWeight()}")
print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
