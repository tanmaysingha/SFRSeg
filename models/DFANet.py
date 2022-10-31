# -*- coding: utf-8 -*-
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Implementation of FANet using Tensorflow 2.1.0 and Keras 2.3.0
"""

import tensorflow as tf
from tensorflow import keras
def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):
  
  if(conv_type == 'ds'):
    x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)
  else:
    x = tf.keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)  
  
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.activations.relu(x)
  
  if (relu):
    x = tf.keras.activations.relu(x)
  
  return x

def enc_block(inputs, kernel, kernel_size, strides, padding='same'):
  x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.activations.relu(x)

  x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides = (1, 1))(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.activations.relu(x)

  x = tf.keras.layers.SeparableConv2D(kernel*4, kernel_size, padding=padding, strides = (1, 1))(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.activations.relu(x)

  #projection
  y = tf.keras.layers.Conv2D(kernel*4, 3, padding=padding, strides = strides)(inputs)
  y = tf.keras.layers.BatchNormalization()(y)
  y = tf.keras.activations.relu(y)
  projection = tf.keras.layers.add([x, y])
  return projection

def enc(inputs, filters, kernel, strides, n):
  x = enc_block(inputs, filters, kernel, strides)
  
  for i in range(1, n):
    x = enc_block(x, filters, kernel, 1)

  return x

def attention(inputs, filters):
    x= tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(1000, activation=None)(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dense(filters, activation=None)(x)
    x = tf.keras.activations.sigmoid(x)
    x = tf.keras.layers.Multiply()([x, inputs])
    x = tf.keras.layers.Conv2D(192, 1, 1, padding='same', activation=None)(x)
    return x

def model(num_classes=20, input_size=(1024, 2048, 3)):

  # Input Layer
  input_layer = tf.keras.layers.Input(shape=input_size, name = 'input_layer')

  ## Step 1: Learning to DownSample
  conv1 = conv_block(input_layer, 'conv', 8, (3, 3), strides = (2, 2))
  
  branch0_enc2 = enc(conv1, 12, (3, 3), strides=2, n = 4) 
  branch0_enc3 = enc(branch0_enc2, 24, (3, 3), strides=2, n = 6) 
  branch0_enc4 = enc(branch0_enc3, 48, (3, 3), strides=2, n = 4) 
    
  atten0 = attention(branch0_enc4, 192)
  concat0 = tf.keras.layers.UpSampling2D((4, 4))(atten0)
  #concat0 = tf.keras.layers.DepthwiseConv2D((1,1), strides=(1, 1), depth_multiplier=1, dilation_rate=4, padding='same')(concat0)
  concat0 = tf.keras.layers.Conv2D(192, 1, 1, padding='same', activation=None)(concat0)
  concat0 = tf.keras.layers.BatchNormalization()(concat0)
  concat0 = tf.keras.layers.Concatenate()([branch0_enc2, concat0])
  
  branch1_enc2 = enc(concat0, 12, (3, 3), strides=2, n = 4) 
  branch1_enc3 = enc(branch1_enc2, 24, (3, 3), strides=2, n = 6) 
  branch1_enc4 = enc(branch1_enc3, 48, (3, 3), strides=2, n = 4)  
    
  atten1 = attention(branch1_enc4, 192)
  concat1 = tf.keras.layers.UpSampling2D((4, 4))(atten1)
  concat1 = tf.keras.layers.Conv2D(192, 1, 1, padding='same', activation=None)(concat1)
  concat1 = tf.keras.layers.BatchNormalization()(concat1)
  concat1 = tf.keras.layers.Concatenate()([branch1_enc2, concat1])

  branch2_enc2 = enc(concat1, 12, (3, 3), strides=2, n = 4) 
  branch2_enc3 = enc(branch2_enc2, 24, (3, 3), strides=2, n = 6) 
  branch2_enc4 = enc(branch2_enc3, 48, (3, 3), strides=2, n = 4)
  
  atten2 = attention(branch2_enc4, 192)
  concat2 = tf.keras.layers.UpSampling2D((4, 4))(atten2)
  concat2 = tf.keras.layers.Conv2D(192, 1, 1, padding='same', activation=None)(concat2)
  concat2 = tf.keras.layers.BatchNormalization()(concat2)
  concat2 = tf.keras.layers.Concatenate()([branch2_enc2, concat2])
    
  #Decoder design
  branch2_enc2_up = tf.keras.layers.UpSampling2D((4, 4))(branch2_enc2)
  branch2_enc2_up = tf.keras.layers.Conv2D(64, 3, 1, padding='same', activation=None)(branch2_enc2_up)
  branch2_enc2_up = tf.keras.layers.BatchNormalization()(branch2_enc2_up)
    
  branch1_enc2_up = tf.keras.layers.UpSampling2D((2, 2))(branch1_enc2)
  branch1_enc2_up = tf.keras.layers.Conv2D(64, 3, 1, padding='same', activation=None)(branch1_enc2_up)
  branch1_enc2_up = tf.keras.layers.BatchNormalization()(branch1_enc2_up)

  branch0_enc2_up = tf.keras.layers.Conv2D(64, 3, 1, padding='same', activation=None)(branch0_enc2)
  branch0_enc2_up = tf.keras.layers.BatchNormalization()(branch0_enc2_up)

  x_shallow = tf.keras.layers.add([branch0_enc2_up, branch1_enc2_up, branch2_enc2_up])
  x_shallow = tf.keras.layers.Conv2D(64, 3, 1, padding='same', activation=None)(x_shallow)
  x_shallow = tf.keras.layers.BatchNormalization()(x_shallow)
    
  branch2_enc4_up = tf.keras.layers.UpSampling2D((16, 16))(atten2)
  branch2_enc4_up = tf.keras.layers.Conv2D(64, 3, 1, padding='same', activation=None)(branch2_enc4_up)
  branch2_enc4_up = tf.keras.layers.BatchNormalization()(branch2_enc4_up)
    
  branch1_enc4_up = tf.keras.layers.UpSampling2D((8, 8))(atten1)
  branch1_enc4_up = tf.keras.layers.Conv2D(64, 3, 1, padding='same', activation=None)(branch1_enc4_up)
  branch1_enc4_up = tf.keras.layers.BatchNormalization()(branch1_enc4_up)

  branch0_enc4_up = tf.keras.layers.UpSampling2D((4, 4))(atten0)
  branch0_enc4_up = tf.keras.layers.Conv2D(64, 3, 1, padding='same', activation=None)(branch0_enc4_up)
  branch0_enc4_up = tf.keras.layers.BatchNormalization()(branch0_enc4_up)

  x_deep = tf.keras.layers.add([x_shallow, branch0_enc4_up, branch1_enc4_up, branch2_enc4_up]) 

  classifier = tf.keras.layers.Conv2D(num_classes, 1, 1, padding='same', activation=None,
                                     kernel_regularizer=keras.regularizers.l2(0.00004), 
                                     bias_regularizer=keras.regularizers.l2(0.00004))(x_deep)
  classifier = tf.keras.layers.Dropout(0.3)(classifier)
  classifier = tf.keras.layers.UpSampling2D((4, 4))(classifier)
  #Since its likely that mixed precision training is used, make sure softmax is float32
  classifier = tf.dtypes.cast(classifier, tf.float32)
  classifier = tf.keras.activations.softmax(classifier)

  DFANet = tf.keras.Model(inputs = input_layer , outputs = classifier, name = 'DFANet')

  return DFANet
    