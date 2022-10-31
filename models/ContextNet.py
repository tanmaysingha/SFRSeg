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

### Note: In the following model design, instead of feeding two different sizes input
### we feed one size input and immediate after the first Conv layer, two branches
### deep and shallow are created.


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, ReLU, Input, Add, UpSampling2D
from tensorflow.keras import activations


def conv_block(x, filters, kernel=(3,3), stride=(1,1), do_relu=False): #FIXME does this need relu or anything?
    
    #Single basic convolution
    x = Conv2D(filters=filters, kernel_size=kernel, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)

    if do_relu:
        x = ReLU()(x)

    return x


def depthwise_separable_conv_block(x, filters, kernel=(3, 3), stride=(1,1)):
    
    # Depthwise convolution (one for each channel)
    x = DepthwiseConv2D(kernel_size=kernel, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)

    # Pointwise convolution (1x1) for actual new features
    x = Conv2D(filters, kernel_size=(1,1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x


def bottleneck_res_block_part(x, filters, expansion_factor, kernel, stride):

    # Keeping shortcut to start for eventual joining
    x_shortcut = x
    
    ## Getting number of channels in input x (source: https://github.com/xiaochus/MobileNetV2)
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    # Depth
    initial_channels = keras.backend.int_shape(x)[channel_axis]
    expanded_channels = initial_channels * expansion_factor
    # # Width
    # cchannel = int(filters * alpha) 

    ## EXPANSION LAYER - 1x1 pointwise convolution layer
    x = Conv2D(filters=expanded_channels, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    ## DEPTHWISE LAYER - 3x3 depthwise convolution layer
    x = DepthwiseConv2D(kernel_size=kernel, strides=stride, padding='same')(x) # FIXME this padding seems correct (needed for addition) but I haven't seen any explicit references to it
    x = BatchNormalization()(x) # GH implementation has this: x = BatchNormalization(axis=channel_axis)(x)
    x = ReLU()(x)

    # PROJECTION LAYER - 1x1 pointwise convolution layer
    x = Conv2D(filters=filters, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)

    # RESIDUAL CONNECTION    
    if initial_channels == filters and stride == (1,1): #FIXME GET THIS CHECKED
        x = keras.layers.Add()([x_shortcut, x])

    return x


def bottleneck_res_block(x, filters, expansion_factor, kernel=(3, 3), stride=(1, 1), repeats=1): # sources: https://arxiv.org/abs/1801.04381 https://machinethink.net/blog/mobilenet-v2/#:~:text=The%20default%20expansion%20factor%20is,to%20that%20144%20%2Dchannel%20tensor.

    x = bottleneck_res_block_part(x, filters, expansion_factor, kernel, stride)

    # Performing repetitions (with stride (1,1), guaranteeing residual connection)
    for i in range(1, repeats):
        x = bottleneck_res_block(x, filters, expansion_factor, kernel, (1,1))

    return x

def pyramid_pooling_block(input_tensor, bin_sizes, input_size):
  concat_list = [input_tensor]
  w = input_size[0] // 32
  h = input_size[1] // 32

  for bin_size in bin_sizes:
    x = tf.keras.layers.AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size))(input_tensor)
    x = tf.keras.layers.Conv2D(32, 3, 2, padding='same',kernel_regularizer=keras.regularizers.l2(0.00004), bias_regularizer=keras.regularizers.l2(0.00004))(x)
    x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w,h)))(x)

    concat_list.append(x)

  return tf.keras.layers.concatenate(concat_list)

def model(num_classes=19, input_size=(1024, 2048, 3), shrink_factor=4):
    """
    Returns an instance of the ContextNet model.
    \nParameters:
    \nnum_classes - Integer, number of classes within classification for model. (Default 19)
    \ninput_size - Tuple of 3, dimensionality of model input into shallow branch. (Default (512, 1024, 3))
    """
    ## SHALLOW BRANCH
    
    shallow_input = Input(shape=input_size, name='input_shallow')
    shallow_branch0 = conv_block(shallow_input, filters=32, kernel=(3,3), stride=(2,2))
    downsample = shallow_branch0
    shallow_branch0 = depthwise_separable_conv_block(shallow_branch0, filters=48, kernel=(3,3), stride=(2, 2))
    shallow_branch = depthwise_separable_conv_block(shallow_branch0, filters=64, kernel=(3,3), stride=(2, 2))  
    shallow_branch = depthwise_separable_conv_block(shallow_branch, filters=96, kernel=(3,3), stride=(1, 1))

    ## DEEP BRANCH
    # According to the original paper 
    #reduced_input_size = (int(input_size[0]/shrink_factor), int(input_size[1]/shrink_factor), input_size[2])
    #deep_input = Input(shape=reduced_input_size, name='input_deep')

    # Small modification is made here: instead of lower resolution direct input, 
    # we feed the output of first Conv_block (shallow_branch0) which has half of
    # of the original RGB input image.

    deep_branch = conv_block(shallow_branch0, filters=32, kernel=(3,3), stride=(2,2), do_relu=True)
    deep_branch = bottleneck_res_block(deep_branch, filters=24, expansion_factor=1, repeats=1) 
    deep_branch = bottleneck_res_block(deep_branch, filters=32, expansion_factor=6, repeats=1)
    deep_branch = bottleneck_res_block(deep_branch, filters=48, expansion_factor=6, repeats=3, stride=(2,2))
    deep_branch = bottleneck_res_block(deep_branch, filters=64, expansion_factor=6, repeats=3, stride=(2,2)) 
    deep_branch = bottleneck_res_block(deep_branch, filters=96, expansion_factor=6, repeats=2)
    deep_branch = bottleneck_res_block(deep_branch, filters=128, expansion_factor=6, repeats=2)
    #deep_branch = conv_block(deep_branch, filters=128, kernel=(3,3), do_relu=True) 
    
    #PPM
    deep_branch = pyramid_pooling_block(deep_branch, [2,4,6,8], input_size)
    
    deep_branch = conv_block(deep_branch, filters=128, kernel=(1,1), do_relu=True) 
    
    
    ## FEATURE FUSION 
    # Deep branch prep
    deep_branch = UpSampling2D((4, 4))(deep_branch) 
    deep_branch = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(4,4), padding='same')(deep_branch)
    deep_branch = tf.keras.layers.BatchNormalization()(deep_branch) 
    deep_branch = tf.keras.activations.relu(deep_branch) 
    deep_branch = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding='same')(deep_branch) 

    # Shallow branch prep
    shallow_branch = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding='same')(shallow_branch)
    
    # Actual addition
    output = Add()([shallow_branch, deep_branch])
    output = tf.keras.activations.relu(output) 
    #output = tf.keras.layers.Concatenate(axis=3)([shallow_branch, deep_branch])
    
    # Dropout layer before final softmax
    
    output = tf.keras.layers.Dropout(rate=0.35)(output)

    # Final result using number of classes
    output = Conv2D(filters=num_classes, kernel_size=(1,1), strides=(1,1), activation='softmax', name='conv_output')(output)

    output = UpSampling2D((8, 8))(output)

    ## MAKING MODEL
    contextnet = Model(inputs=shallow_input, outputs=output)

    return contextnet