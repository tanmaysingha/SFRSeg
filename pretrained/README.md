## Pretrained weights for SFRSeg

SFRSeg was implemented in [tensorflow](https://www.tensorflow.org/) and [keras](https://keras.io/), and pre-trained weights were saved as hdf5 files. 

Below is the link to a cloud folder that contains pre-trained weights

[https://cloudstor.aarnet.edu.au/plus/s/feEYJjVCGBj5wTU](https://cloudstor.aarnet.edu.au/plus/s/feEYJjVCGBj5wTU)

* Cityscapes pre-trained weight
* KITTY pre-trained weight
* Camvid pre-trained weight
* Indoor objects pre-trained weight

Users not familiar with tensorflow/keras should consult relevant [documentation](https://www.tensorflow.org/guide/keras/save_and_serialize) to understand how to load pre-trained weights into a model.

## Requirements for the project

* TensorFlow 2.1
* CUDA = 10.1
* Horovod 0.19.5
* Python = 3.7

## For train.py and predict.py scripts

For training and model's prediction using pre-trained weight, follow the instructions from the following GitHub repository:
https://github.com/SkyWa7ch3r/ImageSegmentation

Train.py script can utilize multiple GPUs. Hence, if you have multiple GPUs and the required system environment, then you can specify the number of GPUs while running the train.py script.
