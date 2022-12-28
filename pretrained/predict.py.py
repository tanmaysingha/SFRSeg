import SFRSeg
import FANet
import DFANet
import FastScnn
import ICNet

from tensorflow import keras
import tensorflow as tf
import cityscapesscripts.helpers.labels as labels
import numpy as np
import PIL
import argparse
import sys
import os
import glob
import time

gpus = tf.config.list_physical_devices('GPU')
# Set Memory Growth to alleviate memory issues
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus, 'GPU')

#----------IMPORT MODELS AND UTILITIES----------#

#----------CONSTANTS----------#

CITYSCAPES_LABELS = [
    label for label in labels.labels if -1 < label.trainId < 255]
# Add unlabeled
#CITYSCAPES_LABELS.append(labels.labels[0])
# Get the IDS
CITYSCAPES_IDS = [label.id for label in CITYSCAPES_LABELS]

# Get the TRAINIDS
CITYSCAPES_TRAINIDS = [label.trainId for label in CITYSCAPES_LABELS]

# Get all the colors 
CITYSCAPES_COLORS = [label.color for label in CITYSCAPES_LABELS]
CLASSES = 19

#----------ARGUMENTS----------#
parser = argparse.ArgumentParser(
    prog='predict', description="Generate predictions from a batch of images from cityscapes")
parser.add_argument(
    "-m", "--model",
    help="Specify the model you wish to use: OPTIONS: SFRSeg, FANet, DFAnet, ICNet",
    choices=['SFRSeg', 'FANet', 'DFANet',
             'fastscnn', 'ICNet', 'ContextNet', 'STDC1_Seg', 'STDC2_Seg', 'BiseNetV2'],
    required=True)
parser.add_argument(
    "-w", "--weights",
    help="Specify the weights path",
    type=str)
parser.add_argument(
    '-p', "--path",
    help="Specify the root folder for cityscapes dataset, if not used looks for CITYSCAPES_DATASET environment variable",
    type=str)
parser.add_argument(
    '-r', "--results-path",
    help="Specify the path for results",
    type=str)
parser.add_argument(
    '-c', "--coarse",
    help="Use the coarse images", action="store_true")
parser.add_argument(
    '-t', "--target-size",
    help="Set the image size for training, should be a elements of a tuple x,y,c",
    default=(1024, 2048, 3),
    type=tuple)
parser.add_argument(
    '--backbone',
    help="The backbone for the deeplabv3+ model",
    choices=['mobilenetv2', 'xception']
)

args = parser.parse_args()
# Get model_name
model_name = args.model
# Check the CITYSCAPES_ROOT path
if os.path.isdir(args.path):
    CITYSCAPES_ROOT = args.path
elif 'CITYSCAPES_DATASET' in os.environ:
    CITYSCAPES_ROOT = os.environ.get('CITYSCAPES_DATASET')
else:
    parser.error("ERROR: No valid path for Cityscapes Dataset given")
# Now do the target size
if args.target_size is None:
    target_size = (1024, 2048, 3)
else:
    target_size = args.target_size
# Check the path of weights
if not os.path.isfile(args.weights):
    parser.error("ERROR: Weights File not found.")
# Ensure results dir is made
if args.results_path is not None:
    if not os.path.isdir(args.results_path):
        os.makedirs(args.results_path)
else:
    parser.error("No results path specified")

files = datasets.get_cityscapes_files(CITYSCAPES_ROOT, 'leftImg8bit', 'test', 'leftImg8bit')

# Get the model
if model_name == 'SFRSeg':
    model = SFRSeg.model(input_size=target_size, num_classes=CLASSES)
if model_name == 'FANet':
    model = FANet.model(input_size=target_size, num_classes=CLASSES)
elif model_name == 'DFANet':
    model = DFANet.model(num_classes=CLASSES, input_size=target_size)
elif model_name == 'ICNet':
    model = ICNet.model(input_size=target_size, num_classes=CLASSES)    
elif model_name == 'ContextNet':
    model = ContextNet.model(input_size=target_size, num_classes=CLASSES)    
elif model_name == 'FastScnn':
    model = FastScnn.model(input_size=target_size, num_classes=CLASSES)    
elif model_name == 'STDC1_Seg':
    model = STDC1_Seg.model(input_size=target_size, num_classes=CLASSES)    
elif model_name == 'STDC2_Seg':
    model = STDC2_Seg.model(input_size=target_size, num_classes=CLASSES)    
elif model_name == 'BiseNetV2':
    model = BiseNetV2.model(input_size=target_size, num_classes=CLASSES)

# Load the weights
model.load_weights(args.weights)
total = 0
elapsed_times = []
for image_path in files:
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_png(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float64)
    image = tf.image.resize(image, (args.target_size[0], args.target_size[1]), method='nearest')
    image = tf.expand_dims(image, 0)

    prediction = model.predict(image)

    prediction = tf.reshape(tf.argmax(prediction, axis=-1),
                            (target_size[0], target_size[1], 1))
    prediction = np.matmul(tf.one_hot(prediction, CLASSES), CITYSCAPES_IDS)
    #if model is trained with different size and want to genarate 1024x2048 size output
    #prediction = tf.image.resize(prediction, (1024, 2048), method='nearest')

    semantic_map = []
    for id in CITYSCAPES_TRAINIDS:
      class_map = tf.reduce_all(tf.equal(prediction, id), axis=-1)
      semantic_map.append(class_map)
    # Save the Image
    semantic_map = tf.stack(semantic_map, axis=-1)
    semantic_map = tf.cast(semantic_map, tf.float32)
    prediction = semantic_map.numpy()
    prediction = np.matmul(prediction, CITYSCAPES_COLORS)
        
    pil = tf.keras.preprocessing.image.array_to_img(
        prediction, data_format="channels_last", scale=False)
    pil.save(os.path.join(args.results_path, os.path.basename(image_path)))
