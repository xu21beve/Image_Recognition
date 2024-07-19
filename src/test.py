import cv2

from PIL import Image
import numpy as np
from tflite_runtime.interpreter import Interpreter

model_path = '../detect.tflite'
img_path = '../selfie.png'


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


  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  
  # Get all outputs from the model
  boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
  # classes = interpreter.get_tensor(output_details[0]['index'])[1] # Class index of detected objects
  # scores = interpreter.get_tensor(output_details[0]['index'])[2] # Confidence of detected objects
  # # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

  # results = []
  # for i in range(count):
    # if scores[i] >= threshold:
      # result = {
        # 'bounding_box': boxes[i],
        # 'class_id': classes[i],
        # 'score': scores[i]
      # }
      # results.append(result)

  # start_time = time.time()
  # interpreter.invoke()
  # stop_time = time.time()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  # results = np.squeeze(output_data)

  '''
  ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            '''
  # top_k = results.argsort()[-5:][::-1]
  # labels = load_labels(args.label_file)
  print(boxes)
  return boxes


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
  original_image_np = np.array(original_image, dtype=np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj[0]
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
