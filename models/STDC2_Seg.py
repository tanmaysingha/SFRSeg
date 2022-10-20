#STDC2 semantic segmentation model without detail guidance

# -*- coding: utf-8 -*-


from functools import reduce
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow.keras.backend as K


#### Custom function for conv2d: conv_block
def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same'):
  
      if(conv_type == 'ds'):
        x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)
      else:
        x = tf.keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)  

      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.activations.relu(x)

      return x

def STDC_module(inputs, filters, kernel, s):
    
    
    tchannel1 = filters
    tchannel2 = tchannel1 // 2
    tchannel3 = tchannel1 // 4
    tchannel4 = tchannel1 // 8

    if s==2:
        x1 = conv_block(inputs, 'conv', tchannel2, (1, 1), strides=(1, 1)) #
        x2 = conv_block(x1, 'conv', tchannel3, (3, 3), strides=(s, s))
        x3 = conv_block(x2, 'conv', tchannel4, (3, 3), strides=(1, 1))
        x4 = conv_block(x3, 'conv', tchannel4, (3, 3), strides=(1, 1))

        #Used MaxPooling instead of average pooling
        x1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x1)
    
    else:
        x1 = conv_block(inputs, 'conv', tchannel2, (1, 1), strides=(1, 1))
        x2 = conv_block(x1, 'conv', tchannel3, (3, 3), strides=(1, 1))
        x3 = conv_block(x2, 'conv', tchannel4, (3, 3), strides=(1, 1))
        x4 = conv_block(x3, 'conv', tchannel4, (3, 3), strides=(1, 1))

    
    
    concat = tf.keras.layers.Concatenate()([x1, x2, x3, x4])

    return concat


"""#### STDC_block to control the repetition of STDC modules"""

def STDC_block(inputs, filters, kernel, strides, n):
      x = STDC_module(inputs, filters, kernel, strides)

      for i in range(1, n):
        x = STDC_module(x, filters, kernel, 1)

      return x    

#Feature Fusion Module
def ffm_block(f1, f2, kernel=256):
    ffm = tf.keras.layers.Concatenate()([f1, f2]) #256+128 = 384
    ffm = conv_block(ffm, 'conv', kernel, (1, 1), strides = (1, 1))
    atten = tf.nn.avg_pool2d(ffm, kernel, 1, 'SAME')
    atten = tf.keras.layers.Conv2D(kernel, 1, padding='same', strides = 1)(atten)
    atten = tf.keras.activations.relu(atten)
    
    atten = tf.keras.layers.Conv2D(kernel, 1, padding='same', strides = 1)(atten)
    atten = tf.keras.activations.sigmoid(atten)
    ffm_atten = tf.keras.layers.Multiply()([atten, ffm])
    ffm_out = tf.keras.layers.add([ffm_atten, ffm])
    
    return ffm_out
    
# Attention Refinement Module
def arm_block(inputs, kernel):
  
      #tchannel = tf.keras.backend.int_shape(inputs)[-1] * 2 
      x = conv_block(inputs, 'conv', kernel, (3, 3), strides = (1, 1))
      x = tf.nn.avg_pool2d(x, kernel, 1, 'SAME')
      x = tf.keras.layers.Conv2D(kernel, 1, padding='same', strides = 1)(x)
      x = conv_block(x, 'conv', kernel, (1, 1), strides = (1, 1))  
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.activations.sigmoid(x)
      x = tf.keras.layers.Multiply()([x, inputs])

      return x   
    

def model(num_classes=19, input_size=(1024, 2048, 3)):

      # Input Layer
      input_layer = tf.keras.layers.Input(shape=input_size, name = 'input_layer')

      ## Step 1: Learning to DownSample
      convx1 = conv_block(input_layer, 'conv', 32, (3, 3), strides = (2, 2))

      convx2 = conv_block(convx1, 'conv', 64, (3, 3), strides = (2, 2))

      #STDC blocks  
      D3 = STDC_block(convx2, 256, (3, 3), strides=2, n=4)

      D4 = STDC_block(D3, 512, (3, 3), strides=2, n=5)

      D5 = STDC_block(D4, 1024, (3, 3), strides=2, n=3)
        
      D5 = tf.nn.avg_pool2d(D5, 1024, 1, 'SAME')
      D5 = conv_block(D5, 'conv', 128, (1, 1), strides = (1, 1))
      D5 = tf.keras.layers.UpSampling2D((1, 1))(D5)
    
      D5_arm = arm_block(D5, 128)
      D5_arm = tf.keras.layers.add([D5_arm, D5])
      D5_up = tf.keras.layers.UpSampling2D((2, 2))(D5_arm)
      D5_up = conv_block(D5_up, 'conv', 128, (3, 3), strides = (1, 1))
      
      D4 = conv_block(D4, 'conv', 128, (1, 1), strides = (1, 1))
      D4_arm = arm_block(D4, 128)
      D4_arm = tf.keras.layers.add([D4_arm, D5_up])
      D4_up = tf.keras.layers.UpSampling2D((2, 2))(D4_arm)
      D4_up = conv_block(D4_up, 'conv', 256, (3, 3), strides = (1, 1))
    
      #feature fusion module
      ffm = ffm_block(D4_up, D3, 256)
      
      #classifier
      classifier = conv_block(ffm, 'conv', 256, (3, 3), strides = (1, 1))
      classifier = tf.keras.layers.Conv2D(num_classes, 1, padding='same', strides = 1)(classifier)
        
      classifier = tf.keras.layers.Dropout(0.3)(classifier)
      classifier = tf.keras.layers.UpSampling2D((8, 8))(classifier)
      classifier = tf.dtypes.cast(classifier, tf.float32)
      classifier = tf.keras.activations.softmax(classifier)

      STDC2_seg = tf.keras.Model(inputs = input_layer , outputs = classifier, name = 'STDC2_seg')

      return STDC2_seg