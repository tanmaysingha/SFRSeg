# New design of a DCNN model (contextNet style) (Moritz and Tanmay version-3/6/21)
# Experimental - to be polished

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
def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
  f1 = tf.keras.layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                       use_bias=True, dilation_rate=2,name=f'{name}/conv')
    
  f2 = tf.keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/bn')
  #f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
  return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

def DCM_block(input_tensor):
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

  # Separable Convolution
  conv3x3_d6 = tf.keras.layers.SeparableConv2D(ch, 3, strides=1, padding='same', dilation_rate=(6, 6))(relu1x1)
  norm3x3_d6 = tf.keras.layers.BatchNormalization()(conv3x3_d6)
  relu3x3_d6 = tf.keras.activations.relu(norm3x3_d6)
  
  B2_feed = tf.keras.layers.add([relu1x1, relu3x3_d6])
  B2_feed = tf.keras.activations.relu(B2_feed)

  # Separable Convolution
  conv3x3_d12 = tf.keras.layers.SeparableConv2D(ch, 3, strides=1, padding='same', dilation_rate=(12, 12))(B2_feed)
  norm3x3_d12 = tf.keras.layers.BatchNormalization()(conv3x3_d12)
  relu3x3_d12 = tf.keras.activations.relu(norm3x3_d12)
  
  B3_feed = tf.keras.layers.add([relu1x1, relu3x3_d12])
  B3_feed = tf.keras.activations.relu(B3_feed)
  
  # Separable Convolution
  conv3x3_d18 = tf.keras.layers.SeparableConv2D(ch, 3, strides=1, padding='same', dilation_rate=(18, 18))(B3_feed)
  norm3x3_d18 = tf.keras.layers.BatchNormalization()(conv3x3_d18)
  relu3x3_d18 = tf.keras.activations.relu(norm3x3_d18)


  concat = tf.keras.layers.Concatenate()([relu1x1, relu3x3_d6, relu3x3_d12, relu3x3_d18])

  return concat


def model(num_classes=19, input_size=(1024, 2048, 3)):
    """
    Returns an instance of the ContextNet model.
    \nParameters:
    \nnum_classes - Integer, number of classes within classification for model. (Default 19)
    \ninput_size - Tuple of 3, dimensionality of model input into shallow branch. (Default (512, 1024, 3))
    """
    ## DEEP BRANCH
    # Reference: Table on page 5 of paper
    #reduced_input_size = (int(input_size[0]/shrink_factor), int(input_size[1]/shrink_factor), input_size[2]) # Reducing input size in deep branch
    
    #down-sampling
    shallow_input = tf.keras.layers.Input(shape=input_size, name='input_shallow')
    conv1 = conv_block(shallow_input, 'conv', 24, (3, 3), strides = (2, 2), dilation_rate=(1, 1)) #512x1024x24 shallow_branch0
    ds1 = conv_block(conv1, 'ds', 32, (3, 3), strides = (2, 2), dilation_rate=(1, 1)) #256x512x32
    
    #deep branch starts from here
    #deep_input = tf.keras.layers.Input(shape=reduced_input_size, name='input_deep')    
    deep_branch1 = bottleneck_block(ds1, 32, (3, 3), t = 1, strides = 2, dilation_rate=(1, 1), n = 1) #128x256x32
    deep_branch1 = bottleneck_block(deep_branch1, 48, (3, 3), t = 6, strides = 1, dilation_rate=(2, 2), n = 2) #128x256x48  
    
    ## SHALLOW BRANCH starts from here
    #shallow_input = tf.keras.layers.Input(shape=input_size, name='input_shallow')
    #shallow_branch0 = conv_block(shallow_input, 'conv', 24, (3, 3), strides = (2, 2), dilation_rate=(1, 1)) #512x1024x24
    #shallow_branch1 = conv_block(conv1, 'ds', 32, (3, 3), strides = (2, 2), dilation_rate=(1, 1)) #256x512x32
    #downsample = shallow_branch
    shallow_branch1 = conv_block(ds1, 'ds', 48, (3, 3), strides = (2, 2), dilation_rate=(1, 1))  #128x256x48
    
    Deep_shallow_Add1 = tf.keras.layers.add([shallow_branch1, deep_branch1])
    Deep_shallow_Add1 = tf.keras.activations.relu(Deep_shallow_Add1)
    Deep_shallow_Add1 = DCM_block(Deep_shallow_Add1) #128x256x48
    
    deep_branch2 = bottleneck_block(Deep_shallow_Add1, 64, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 2) #64x128x64
    deep_branch2 = bottleneck_block(deep_branch2, 80, (3, 3), t = 6, strides = 1, dilation_rate=(2, 2), n = 1)  #64x128x96

    shallow_branch2 = conv_block(Deep_shallow_Add1, 'ds', 80, (3, 3), strides = (2, 2), dilation_rate=(1, 1))  #64x128x96

    Deep_shallow_Add2 = tf.keras.layers.add([shallow_branch2, deep_branch2])
    Deep_shallow_Add2 = tf.keras.activations.relu(Deep_shallow_Add2)
    Deep_shallow_Add2 = DCM_block(Deep_shallow_Add2) #64x128x96

    deep_branch3 = bottleneck_block(Deep_shallow_Add2, 96, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 1) #32x64x128


    shallow_branch3 = conv_block(Deep_shallow_Add2, 'ds', 96, (3, 3), strides = (2, 2), dilation_rate=(1, 1))  #32x64x160


    Deep_shallow_Add3 = tf.keras.layers.add([shallow_branch3, deep_branch3])
    Deep_shallow_Add3 = tf.keras.activations.relu(Deep_shallow_Add3)
    Deep_shallow_Add3 = DCM_block(Deep_shallow_Add3) #32x64x160
   
    shared_branch = bottleneck_block(Deep_shallow_Add3, 128, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 1)  #16x32x160
    
    #new design
    
    ##Step3: Bottleneck blocks ( 2nd encoder)
    deep_branch22 = tf.keras.layers.UpSampling2D(size=(8,8), interpolation='bilinear')(shared_branch)
    deep_branch22 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(8,8), padding='same')(deep_branch22)
    deep_branch22 = tf.keras.layers.BatchNormalization()(deep_branch22)
    #output = tf.keras.activations.swish(output)
    deep_branch22 = tf.keras.layers.Activation('relu')(deep_branch22)
    #deep_branch22 = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(deep_branch22)
    deep_branch22 = tf.keras.layers.Conv2D(48, kernel_size=(1,1), strides=(1,1), padding='same')(deep_branch22)
    
    #feature adding from 1st encoder
    Deep_shallow_Add21 = tf.keras.layers.Add()([deep_branch22, Deep_shallow_Add1])
    #P5_out = tf.keras.activations.swish(P5_out)
    Deep_shallow_Add21 = tf.keras.layers.Activation('relu')(Deep_shallow_Add21)
    Deep_shallow_Add21 = DCM_block(Deep_shallow_Add21)
    
    deep_branch32 = bottleneck_block(Deep_shallow_Add21, 64, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 3)
    deep_branch32 = bottleneck_block(deep_branch32, 80, (3, 3), t = 6, strides = 1, dilation_rate=(2, 2), n = 1)
    #feature adding from 1st encoder
    Deep_shallow_Add22 = tf.keras.layers.Add()([deep_branch32, Deep_shallow_Add2])
    Deep_shallow_Add22 = tf.keras.activations.relu(Deep_shallow_Add22)
    Deep_shallow_Add22 = DCM_block(Deep_shallow_Add22)
    
    deep_branch42 = bottleneck_block(Deep_shallow_Add22, 96, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 1) #32x64x128
    #deep_branch42 = bottleneck_block(deep_branch42, 160, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 1)  #32x64x160
    
    Deep_shallow_Add23 = tf.keras.layers.add([Deep_shallow_Add3, deep_branch42])
    Deep_shallow_Add23 = tf.keras.activations.relu(Deep_shallow_Add23)
    Deep_shallow_Add23 = DCM_block(Deep_shallow_Add23) #32x64x160

    shared_branch2 = bottleneck_block(Deep_shallow_Add23, 128, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 1)  #16x32x160
    
    shared_branch2 = tf.keras.layers.add([shared_branch2, shared_branch])
    shared_branch2 = tf.keras.activations.relu(shared_branch2)
    
    ##Step3: Bottleneck blocks ( 3rd encoder)
    deep_branch33 = tf.keras.layers.UpSampling2D(size=(4,4), interpolation='bilinear')(shared_branch2)
    deep_branch33 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(4,4), padding='same')(deep_branch33)
    deep_branch33 = tf.keras.layers.BatchNormalization()(deep_branch33)
    #output = tf.keras.activations.swish(output)
    deep_branch33 = tf.keras.layers.Activation('relu')(deep_branch33)
    #deep_branch33 = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(deep_branch33)
    deep_branch33 = tf.keras.layers.Conv2D(80, kernel_size=(1,1), strides=(1,1), padding='same')(deep_branch33)
    
    #feature adding from 1st encoder
    Deep_shallow_Add32 = tf.keras.layers.Add()([deep_branch33, Deep_shallow_Add22, Deep_shallow_Add2])
    Deep_shallow_Add32 = tf.keras.layers.Activation('relu')(Deep_shallow_Add32)
    Deep_shallow_Add32 = DCM_block(Deep_shallow_Add32)
    
    deep_branch43 = bottleneck_block(Deep_shallow_Add32, 96, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 2) #32x64x128
    #deep_branch43 = bottleneck_block(deep_branch43, 160, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 1)  #32x64x160
    
    Deep_shallow_Add33 = tf.keras.layers.add([deep_branch43, Deep_shallow_Add23, Deep_shallow_Add3])
    Deep_shallow_Add33 = tf.keras.activations.relu(Deep_shallow_Add3)
    Deep_shallow_Add33 = DCM_block(Deep_shallow_Add33) #32x64x160

    shared_branch3 = bottleneck_block(Deep_shallow_Add33, 128, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 1)  #16x32x160
    
    shared_branch3 = tf.keras.layers.add([shared_branch3, shared_branch2, shared_branch])
    shared_branch3 = tf.keras.activations.relu(shared_branch3)
    
    #Bottleneck block (4th encoder)
    deep_branch44 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(shared_branch3)
    deep_branch44 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(2,2), padding='same')(deep_branch44)
    deep_branch44 = tf.keras.layers.BatchNormalization()(deep_branch44)
    #output = tf.keras.activations.swish(output)
    deep_branch44 = tf.keras.layers.Activation('relu')(deep_branch44)
    deep_branch44 = tf.keras.layers.Conv2D(96, kernel_size=(1,1), strides=(1,1), padding='same')(deep_branch44)
    
    Deep_shallow_Add43 = tf.keras.layers.add([deep_branch44, Deep_shallow_Add33, Deep_shallow_Add23, Deep_shallow_Add3])
    Deep_shallow_Add43 = tf.keras.activations.relu(Deep_shallow_Add43)
    Deep_shallow_Add43 = DCM_block(Deep_shallow_Add43) #32x64x160        

    #shared_branch = bottleneck_block(Deep_shallow_Add3, 160, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 1)
    shared_branch4 = bottleneck_block(Deep_shallow_Add43, 128, (3, 3), t = 6, strides = 2, dilation_rate=(1, 1), n = 2)  #16x32x160
    
    shared_branch4 = tf.keras.layers.add([shared_branch4, shared_branch3, shared_branch2, shared_branch])
    shared_branch4 = tf.keras.activations.relu(shared_branch4)
    #shared_branch = DCM_block(shared_branch, 2, 4, 6)

    w = 64 #channel width of DIS-CAM
    #shared features
    F6 = shared_branch4 #16
    F5 = Deep_shallow_Add43 #32
    F4 = Deep_shallow_Add32 #64
    #shallow features
    F3 = Deep_shallow_Add21  #128
    F2 = ds1 #256
    F1 = conv1

    F6_in = tf.keras.layers.Conv2D(w, 1, 1, padding='same', activation=None, name='F6/conv2d')(F6)
    F6_in = tf.keras.layers.BatchNormalization()(F6_in)
    #F6_in = DCM_block(F6_in)
    F7_in = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(F6_in)
    F5_in = tf.keras.layers.Conv2D(w, 1, 1, padding='same', activation=None, name='F5/conv2d')(F5)
    F5_in = tf.keras.layers.BatchNormalization()(F5_in)
    F4_in = tf.keras.layers.Conv2D(w, 1, 1, padding='same', activation=None, name='F4/conv2d')(F4)
    F4_in = tf.keras.layers.BatchNormalization()(F4_in)

    #D2_in = tf.keras.layers.Conv2D(64, 1, 1, padding='same', activation=None, name='D2/conv2d')(D2)
    #D2_in = tf.keras.layers.BatchNormalization()(D2_in)

    F3_in = tf.keras.layers.Conv2D(w, 1, 1, padding='same', activation=None, name='F3/conv2d')(F3)
    F3_in = tf.keras.layers.BatchNormalization()(F3_in)
    F2_in = tf.keras.layers.Conv2D(w, 1, 1, padding='same', activation=None, name='F2/conv2d')(F2)
    F2_in = tf.keras.layers.BatchNormalization()(F2_in)
    F1_in = tf.keras.layers.Conv2D(w, 1, 1, padding='same', activation=None, name='F1/conv2d')(F1)
    F1_in = tf.keras.layers.BatchNormalization()(F1_in)
    #shallow_branch2 = tf.keras.layers.Conv2D(64, 1, 1, padding='same', activation=None, name='SB2')(shallow_branch2)
    #shallow_branch2 = tf.keras.layers.BatchNormalization()(shallow_branch2)

    #Top-down
    F7_U = tf.keras.layers.UpSampling2D()(F7_in)
    F6_td = tf.keras.layers.Add()([F6_in, F7_U])
    F6_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F6_td)
    #F6_td = tf.keras.activations.swish(F6_td) # tensorflow does not have swish
    F6_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1, name='F6_td1')(F6_td)
    
    F6_U = tf.keras.layers.UpSampling2D((2, 2))(F6_td)  #64x128x160   
    F5_td = tf.keras.layers.add([F6_U, F5_in])
    #F4_td = tf.keras.activations.relu(F4_td) #64x128x96
    F5_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F5_td)
    #F5_td = tf.keras.activations.swish(F5_td)
    F5_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F5_td1')(F5_td)

    F5_U = tf.keras.layers.UpSampling2D((2, 2))(F5_td)  #64x128x160   
    F4_td = tf.keras.layers.add([F5_U, F4_in])
    #F4_td = tf.keras.activations.relu(F4_td) #64x128x96
    F4_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F4_td)
    #F4_td = tf.keras.activations.swish(F4_td)
    F4_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F4_td1')(F4_td)
       
    F1_M = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='F1/MaxPool')(F1_in)
    F2_td = tf.keras.layers.add([F1_M, F2_in])
    F2_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F2_td)
    #F2_td = tf.keras.activations.swish(F2_td)
    F2_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F2_bu1')(F2_td)

    F2_M = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='F2/MaxPool')(F2_td)
    F3_td = tf.keras.layers.add([F2_M, F3_in])
    F3_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F3_td)
    #F3_td = tf.keras.activations.swish(F3_td)
    F3_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F3_bu1')(F3_td)

    #Bottom-Up continue
    F3_M = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='F3/MaxPool')(F3_td)
    F4_td = tf.keras.layers.add([F3_M, F4_td])
    F4_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F4_td)
    #F4_td = tf.keras.activations.swish(F4_td)
    F4_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F4_bu1')(F4_td)
    
    F4_td = DCM_block(F4_td)

    F4_M = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='F4/MaxPool')(F4_td)
    F5_td = tf.keras.layers.add([F4_M, F5_td, F5_in])
    F5_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F5_td)
    #F5_td = tf.keras.activations.swish(F5_td)
    F5_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F5_bu1')(F5_td)

    F5_M = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='F5/MaxPool')(F5_td)
    F6_td = tf.keras.layers.add([F5_M, F6_in])
    F6_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F6_td)
    #F6_td = tf.keras.activations.swish(F6_td)
    F6_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F6_bu1')(F6_td)
    
    F6_M = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='F6/MaxPool')(F6_td)
    F7_td = tf.keras.layers.add([F6_M, F7_in])
    F7_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F7_td)
    #F7_td = tf.keras.activations.swish(F7_td)
    F7_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F7_bu1')(F7_td)

    #Top-down

    F7_U = tf.keras.layers.UpSampling2D((2, 2))(F7_td)
    F6_td = tf.keras.layers.add([F7_U, F6_td, F6_in])
    F6_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F6_td)
    #F6_td = tf.keras.activations.swish(F6_td)
    F6_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F6_td2')(F6_td)
    
    F6_U = tf.keras.layers.UpSampling2D((2, 2))(F6_td)  #64x128x160   
    F5_td = tf.keras.layers.add([F6_U, F5_td, F5_in])
    F5_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F5_td)
    #F5_td = tf.keras.activations.swish(F5_td)
    F5_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F5_td2')(F5_td)

    F5_U = tf.keras.layers.UpSampling2D((2, 2))(F5_td)  #64x128x160   
    F4_td = tf.keras.layers.add([F5_U, F4_td, F4_in])
    #F4_td = tf.keras.activations.relu(F4_td) #64x128x96
    F4_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F4_td)
    #F4_td = tf.keras.activations.swish(F4_td)
    F4_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F4_td2')(F4_td)

    F4_U = tf.keras.layers.UpSampling2D((2, 2))(F4_td)  #64x128x160   
    F3_td = tf.keras.layers.add([F4_U, F3_td, F3_in])
    #F4_td = tf.keras.activations.relu(F4_td) #64x128x96
    F3_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F3_td)
    #F3_td = tf.keras.activations.swish(F3_td)
    F3_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F3_td2')(F3_td)

    F3_U = tf.keras.layers.UpSampling2D((2, 2))(F3_td)  #64x128x160   
    F2_td = tf.keras.layers.add([F3_U, F2_td, F2_in])
    #F4_td = tf.keras.activations.relu(F4_td) #64x128x96
    F2_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F2_td)
    #F2_td = tf.keras.activations.swish(F2_td)
    F2_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F2_td2')(F2_td)
    
    F2_U = tf.keras.layers.UpSampling2D((2, 2))(F2_td)  #64x128x160   
    F1_td = tf.keras.layers.add([F2_U, F1_in])
    #F4_td = tf.keras.activations.relu(F4_td) #64x128x96
    F1_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(F1_td)
    #F1_td = tf.keras.activations.swish(F1_td)
    F1_td = SeparableConvBlock(num_channels=w, kernel_size=3, strides=1,name='F1_td2')(F1_td)

    #output module- it is required to add this module (tanmay)
    output = tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same', strides = (1, 1), dilation_rate=4, name = 'DSConv1_output')(F1_td)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.activations.relu(output)

    output = tf.keras.layers.SeparableConv2D(48, (3, 3), padding='same', strides = (1, 1), dilation_rate=4, name = 'DSConv2_output')(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.activations.relu(output)
    
    #need to upsampling
    output = tf.keras.layers.Dropout(0.35)(output)
    #output = tf.keras.layers.Conv2D(num_classes, 1, 1, padding='same', activation=None)(output)
    
    #need to upsampling
    #output = tf.keras.layers.Dropout(0.4)(output)
    output = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(output)
    output = tf.keras.layers.Conv2D(num_classes, 1, 1, padding='same', activation=None,
                                      kernel_regularizer=keras.regularizers.l2(0.00004),
                                      bias_regularizer=keras.regularizers.l2(0.00004))(output)
  
    #Since its likely that mixed precision training is used, make sure softmax is float32
    output = tf.dtypes.cast(output, tf.float32)
    output = tf.keras.activations.softmax(output)

    SFRSeg = tf.keras.Model(inputs = shallow_input, outputs = output, name = 'SFRSeg')

    return SFRSeg
