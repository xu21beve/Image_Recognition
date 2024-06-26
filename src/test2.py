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

import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(3, 3), dtype=np.uint8)  # len(classes) --> substitute for size because classes not initialized

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='../assets/image.jpg',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='../detect.tflite',
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
  for i in top_k:
    if floating_model:
      print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
    else:
      print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))


import cv2

from PIL import Image
import numpy as np
from tflite_runtime.interpreter import Interpreter

model_path = '../detect.tflite'
img_path = '../assets/image.jpg'


# Load the labels into a list
# classes = ['../labels.txt'] * model.model_spec.config.num_classes
# label_map = model.model_spec.config.label_map
# for label_id, label_name in label_map.as_dict().items():
# classes[label_id-1] = label_name

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(3, 3), dtype=np.uint8)  # len(classes) --> substitute for size because classes not initialized

def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  resized_img = tf.cast(resized_img, dtype=tf.uint8)
  return resized_img, original_image


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""

  # signature_fn = interpreter.get_signature_runner()

  # # Feed the input image to the model
  # output = signature_fn(image)

  # Get all outputs from the model
  boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
  # classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
  # scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
  num = interpreter.get_tensor(interpreter.get_output_tensor_index(0))[0][0]  # Total number of detected objects (inaccurate and not needed)

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i] #,
        # 'class_id': classes[i],
        # 'score': scores[i]
      }
      results.append(result)
  return results


def run_odt_and_draw_results(original_image, processed_image, interpreter, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  # preprocessed_image, original_image = preprocess_image(
      # image_path,
      # (input_height, input_width)
    # )

  # Run object detection on the input image
  results = detect_objects(interpreter, processed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])

    # Draw the bounding box and label on the image
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8



# # INPUT_IMAGE_URL = "https://storage.googleapis.com/cloud-ml-data/img/openimage/3/2520/3916261642_0a504acd60_o.jpg"
# DETECTION_THRESHOLD = 0.3

# TEMP_FILE = '../assets/image.jpg'

# # !wget -q -O $TEMP_FILE # $INPUT_IMAGE_URL
# img = Image.open(TEMP_FILE).resize((512, 512))


# # Load the TFLite model
# interpreter = Interpreter(model_path=model_path)
# interpreter.allocate_tensors()

# # Run inference and draw detection result on the local copy of the original file
# detection_result_image = run_odt_and_draw_results(
    # TEMP_FILE,
    # interpreter,
    # threshold=DETECTION_THRESHOLD
# )

interpreter = Interpreter(
      model_path=model_path)  #experimental_delegates=ext_delegate,     num_threads=args.num_threads
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32

# NxHxWxC, H:1, W:2
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
original_img = Image.open(img_path)
processed_img = original_img.resize((width, height))

# add N dim
input_data = np.expand_dims(processed_img, axis=0)

if floating_model:
  input_data = (np.float32(input_data) - 127.5) / 127.5  #TODO: check what input mean and std are

interpreter.set_tensor(input_details[0]['index'], input_data)
  
detection_result_image = run_odt_and_draw_results(
    original_img,
    processed_img,
    interpreter,
    threshold=.5
)

# Show the detection result
Image.fromarray(detection_result_image)

