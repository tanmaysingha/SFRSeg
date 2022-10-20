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

from functools import reduce
import tensorflow as tf
from tensorflow import keras
import numpy as np

#### Custom function for conv2d: conv_block
def conv_block(inputs, conv_type, kernel, kernel_size, strides, dilation_rate=(1,1), padding='same', relu=True, reg=keras.regularizers.l2(0.00004)):
  
  if(conv_type == 'ds'):
    x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)
  else:
    x = tf.keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides = strides, dilation_rate=dilation_rate, kernel_regularizer=reg, bias_regularizer=reg)(inputs)  
  
  x = tf.keras.layers.BatchNormalization()(x)
  
  if (relu):
    x = tf.keras.activations.relu(x)
  
  return x
  
#### residual custom method
def _res_bottleneck(inputs, filters, kernel, t, s, dilation_rate=(1,1), r=False):
    
    
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), dilation_rate=dilation_rate, depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False) 

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x

"""#### Bottleneck custom method"""

def bottleneck_block(inputs, filters, kernel, t, strides, dilation_rate, n):
  x = _res_bottleneck(inputs, filters, kernel, t, strides, dilation_rate=dilation_rate)
  
  for i in range(1, n):
    x = _res_bottleneck(x, filters, kernel, t, 1, True)

  return x

MOMENTUM = 0.997
EPSILON = 1e-4
def SeparableConvBlock(num_channels, kernel_size, strides, freeze_bn=False):
  f1 = tf.keras.layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                       use_bias=True, dilation_rate=2,)
    
  f2 = tf.keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
  return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

def CMM_block(input_tensor):
  shape = list(input_tensor.shape)
  h = shape[1]
  w = shape[2]
  c = shape[3]
  ch = int(shape[3]/4)

  #feature reduction
  # 1x1 Convolution
  conv1x1 = tf.keras.layers.Conv2D(ch, 1, strides=1, padding='same', use_bias=False)(input_tensor)
  norm1x1 = tf.keras.layers.BatchNormalization()(conv1x1)
  relu1x1 = tf.keras.activations.relu(norm1x1)

  conv3x3_d6 = tf.keras.layers.SeparableConv2D(ch, 3, strides=1, padding='same', dilation_rate=(6, 6))(input_tensor)
  norm3x3_d6 = tf.keras.layers.BatchNormalization()(conv3x3_d6)
  relu3x3_d6 = tf.keras.activations.relu(norm3x3_d6)

  # Instead of Image pooling branch separable convolution branch is used
  """pool1 = tf.keras.layers.AveragePooling2D(pool_size=(h, w))(input_tensor)
  conv1 = tf.keras.layers.Conv2D(ch, 1, strides=1, padding='same', use_bias=False)(pool1)
  norm1 = tf.keras.layers.BatchNormalization()(conv1)
  relu1 = tf.keras.layers.Activation('relu')(norm1)
  upsampling1 = tf.keras.layers.UpSampling2D(size=(h, w), interpolation='bilinear')(relu1)"""
  conv3x3_d12 = tf.keras.layers.SeparableConv2D(ch, 3, strides=1, padding='same', dilation_rate=(12, 12))(input_tensor)
  norm3x3_d12 = tf.keras.layers.BatchNormalization()(conv3x3_d12)
  relu3x3_d12 = tf.keras.activations.relu(norm3x3_d12)

  # Instead of Image pooling branch separable convolution branch is used
  """pool2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(input_tensor)
  conv2 = tf.keras.layers.Conv2D(ch, 1, strides=1, padding='same', use_bias=False)(pool2)
  norm2 = tf.keras.layers.BatchNormalization()(conv2)
  relu2 = tf.keras.layers.Activation('relu')(norm2)
  upsampling2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(relu2)"""
  conv3x3_d18 = tf.keras.layers.SeparableConv2D(ch, 3, strides=1, padding='same', dilation_rate=(18, 18))(input_tensor)
  norm3x3_d18 = tf.keras.layers.BatchNormalization()(conv3x3_d18)
  relu3x3_d18 = tf.keras.activations.relu(norm3x3_d18)

  concat = tf.keras.layers.Concatenate()([relu1x1, relu3x3_d6, relu3x3_d12, relu3x3_d18])

  return concat


def model(num_classes=19, input_size=(1024, 2048, 3)):
    
    ## SHALLOW BRANCH
    shallow_input = tf.keras.layers.Input(shape=input_size, name='input_shallow')
    shallow_branch0 = conv_block(shallow_input, 'conv', 24, (3, 3), strides = (2, 2), dilation_rate=(1, 1)) 
    

    ## DEEP BRANCH
    deep_branch1 = conv_block(shallow_branch0, 'conv', 32, (3, 3), strides = (2, 2), dilation_rate=(1, 1)) 
    deep_branch2 = bottleneck_block(deep_branch1, 32, (3, 3), t = 1, strides = 2, dilation_rate=(1, 1), n = 1) 
    deep_branch2 = bottleneck_block(deep_branch2, 48, (3, 3), t = 6, strides = 1, dilation_rate=(1, 1), n = 2)  
    
    shallow_branch1 = conv_block(shallow_branch0, 'ds', 32, (3, 3), strides = (2, 2), dilation_rate=(1, 1)) 
    shallow_branch2 = conv_block(shallow_branch1, 'ds', 48, (3, 3), strides = (2, 2), dilation_rate=(1, 1)) 
    
    Deep_shallow_Add1 = tf.keras.layers.add([shallow_branch2, deep_branch2])
    Deep_shallow_Add1 = tf.keras.activations.relu(Deep_shallow_Add1)
    Deep_shallow_Add1 = CMM_block(Deep_shallow_Add1) 
    
    deep_branch3 = bottleneck_block(Deep_shallow_Add1, 64, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 3) 
    deep_branch3 = bottleneck_block(deep_branch3, 96, (3, 3), t = 6, strides = 1, dilation_rate=(1, 1), n = 1) 

    shallow_branch3 = conv_block(Deep_shallow_Add1, 'ds', 96, (3, 3), strides = (2, 2), dilation_rate=(1, 1)) 

    Deep_shallow_Add2 = tf.keras.layers.add([shallow_branch3, deep_branch3])
    Deep_shallow_Add2 = tf.keras.activations.relu(Deep_shallow_Add2)
    Deep_shallow_Add2 = CMM_block(Deep_shallow_Add2)

    deep_branch4 = bottleneck_block(Deep_shallow_Add2, 128, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 3)
    deep_branch4 = bottleneck_block(deep_branch4, 160, (3, 3), t = 6, strides = 1, dilation_rate=(1, 1), n = 1) 

    shallow_branch4 = conv_block(Deep_shallow_Add2, 'ds', 160, (3, 3), strides = (2, 2), dilation_rate=(1, 1))  

    Deep_shallow_Add3 = tf.keras.layers.add([shallow_branch4, deep_branch4])
    Deep_shallow_Add3 = tf.keras.activations.relu(Deep_shallow_Add3)
    Deep_shallow_Add3 = CMM_block(Deep_shallow_Add3) 

    shared_branch = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(Deep_shallow_Add3)

    w = 64
    #shared features
    F6 = shared_branch 
    F5 = Deep_shallow_Add3 
    F4 = Deep_shallow_Add2 
    #shallow features
    F2 = Deep_shallow_Add1  
    F1 = shallow_branch1 
    F0 = shallow_branch0 

    F6_in = tf.keras.layers.Conv2D(w, 1, 1, padding='same', activation=None, name='F6/conv2d')(F6)
    F6_in = tf.keras.layers.BatchNormalization()(F6_in)
    F5_in = tf.keras.layers.Conv2D(w, 1, 1, padding='same', activation=None, name='F5/conv2d')(F5)
    F5_in = tf.keras.layers.BatchNormalization()(F5_in)
    F4_in = tf.keras.layers.Conv2D(w, 1, 1, padding='same', activation=None, name='F4/conv2d')(F4)
    F4_in = tf.keras.layers.BatchNormalization()(F4_in)

    F2_in = tf.keras.layers.Conv2D(w, 1, 1, padding='same', activation=None, name='F2/conv2d')(F2)
    F2_in = tf.keras.layers.BatchNormalization()(F2_in)
    F1_in = tf.keras.layers.Conv2D(w, 1, 1, padding='same', activation=None, name='F1/conv2d')(F1)
    F1_in = tf.keras.layers.BatchNormalization()(F1_in)
    F0_in = tf.keras.layers.Conv2D(w, 1, 1, padding='same', activation=None, name='F0/convd')(F0)
    F0_in = tf.keras.layers.BatchNormalization()(F0_in)

    #Top-down
    F6_U = tf.keras.layers.UpSampling2D((2, 2))(F6_in)   
    F5_td = tf.keras.layers.add([F6_U, F5_in])
    F5_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F5_td)
    F5_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(F5_td)

    F5_U = tf.keras.layers.UpSampling2D((2, 2))(F5_td)  
    F4_td = tf.keras.layers.add([F5_U, F4_in])
    F4_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F4_td)
    F4_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(F4_td)
    
    F4_U = tf.keras.layers.UpSampling2D((2, 2))(F4_td)

    #Bottom-Up
    F0_M = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='F0/MaxPool')(F0_in)
    F1_td = tf.keras.layers.add([F0_M, F1_in])
    F1_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F1_td)
    F1_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(F1_td)

    F1_M = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='F1/MaxPool')(F1_td)
    F2_td = tf.keras.layers.add([F1_M, F2_in])
    F2_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F2_td)
    F2_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(F2_td)

    #Middle stage
    F2_td = tf.keras.layers.add([F4_U, F2_td])
    F2_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F2_td)
    F2_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(F2_td)

    #Bottom-Up continue
    F3_M = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='F3/MaxPool')(F2_td)
    F4_td = tf.keras.layers.add([F3_M, F4_td, F4_in])
    F4_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F4_td)
    F4_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(F4_td)

    F4_M = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='F4/MaxPool')(F4_td)
    F5_td = tf.keras.layers.add([F4_M, F5_td, F5_in])
    F5_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F5_td)
    F5_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(F5_td)

    F5_M = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='F5/MaxPool')(F5_td)
    F6_td = tf.keras.layers.add([F5_M, F6_in])
    F6_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F6_td)
    F6_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(F6_td)

    #Top-down
    F6_U = tf.keras.layers.UpSampling2D((2, 2))(F6_td)    
    F5_td = tf.keras.layers.add([F6_U, F5_td, F5_in]) 
    F5_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F5_td)
    F5_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(F5_td)

    F5_U = tf.keras.layers.UpSampling2D((2, 2))(F5_td)    
    F4_td = tf.keras.layers.add([F5_U, F4_td, F4_in])
    #F4_td = tf.keras.activations.relu(F4_td) 
    F4_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F4_td)
    F4_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(F4_td)

    F4_U = tf.keras.layers.UpSampling2D((2, 2))(F4_td)     
    F2_td = tf.keras.layers.add([F4_U, F2_td, F2_in])
    #F4_td = tf.keras.activations.relu(F4_td) 
    F2_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F2_td)
    F2_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(F2_td)

    F2_U = tf.keras.layers.UpSampling2D((2, 2))(F2_td)     
    F1_td = tf.keras.layers.add([F2_U, F1_td, F1_in])
    F1_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F1_td)
    F1_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(F1_td)

    F1_U = tf.keras.layers.UpSampling2D((2, 2))(F1_td)   
    F0_td = tf.keras.layers.add([F1_U, F0_in])
    F0_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F0_td)
    F0_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(F0_td)

    output = tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same', strides = (1, 1), dilation_rate=1, name = 'DSConv1_output')(F0_td)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.activations.relu(output)

    output = tf.keras.layers.SeparableConv2D(48, (3, 3), padding='same', strides = (1, 1), dilation_rate=2, name = 'DSConv2_output')(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.activations.relu(output)
    #need to upsampling
    output = tf.keras.layers.Dropout(0.45)(output)
    output = tf.keras.layers.UpSampling2D((2, 2))(output)
    
    # Final result using number of classes
    output = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(1,1), strides=(1,1), activation='softmax', name='conv_output')(output)

    ## MAKING MODEL
    SCMNet = tf.keras.Model(inputs=shallow_input, outputs=output)

    return SCMNet
