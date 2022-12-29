## Pretrained weights for SFRSeg

SFRSeg was implemented in [tensorflow](https://www.tensorflow.org/) and [keras](https://keras.io/), and pre-trained weights were saved as hdf5/h5 files. 

* Cityscapes pre-trained weight: [cityscapes_weight.hdf5](https://cloudstor.aarnet.edu.au/plus/s/JO9ij8gZNZ2bLHz/download)
* KITTY pre-trained weight: [kitti_weight.hdf5](https://cloudstor.aarnet.edu.au/plus/s/OvRHlSBfuqEdTt2/download)
* Camvid pre-trained weight: [camvid_weight.h5](https://cloudstor.aarnet.edu.au/plus/s/jqdoxoUyCDkbcm9/download)
* Indoor objects pre-trained weight: [indoor_weight.hdf5](https://cloudstor.aarnet.edu.au/plus/s/4NVp0yZZZYXjUWG/download)

Users not familiar with tensorflow/keras should consult relevant [documentation](https://www.tensorflow.org/guide/keras/save_and_serialize) to understand how to load pre-trained weights into a model.

## Scripts

Use the script [predict.py](https://github.com/tanmaysingha/SFRSeg/blob/main/pretrained/predict.py) to load the pre-trained weight and generate prediction. All Python scripts implementing SFRSeg and other models under [models](https://github.com/tanmaysingha/SFRSeg/tree/main/models) should also be added to PYTHONPATH. Finally, a training script [train.py](https://github.com/tanmaysingha/SFRSeg/blob/main/pretrained/train.py) is also provided for custom training. For further information please visit https://github.com/SkyWa7ch3r/ImageSegmentation

## Requirements 
* TensorFlow 2.1
* CUDA = 10.1
* Python = 3.7
* Horovod 0.19.5 (optional)

