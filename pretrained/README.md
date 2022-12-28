## Pretrained weights for SFRSeg

SFRSeg was implemented in [tensorflow](https://www.tensorflow.org/) and [keras](https://keras.io/), and pre-trained weights were saved as hdf5 files. 

Below is the link to a cloud folder that contains pre-trained weights

[https://cloudstor.aarnet.edu.au/plus/s/feEYJjVCGBj5wTU](https://cloudstor.aarnet.edu.au/plus/s/feEYJjVCGBj5wTU)

* Cityscapes pre-trained weight
* KITTY pre-trained weight
* Camvid pre-trained weight
* Indoor objects pre-trained weight

Users not familiar with tensorflow/keras should consult relevant [documentation](https://www.tensorflow.org/guide/keras/save_and_serialize) to understand how to load pre-trained weights into a model.

## Using train.py

python train.py --help

usage: 
train.py [-h] -m {SFRSeg,FANet,ICNet,fastscnn,STDC1_Seg} 
                [-r RESUME] [-p PATH] [-c] [-t TARGET_SIZE] [-b BATCH]
                [-e EPOCHS] [--mixed-precision]
                [-f {adadelta,adagrad,adam,adamax,ftrl,nadam,rmsprop,sgd,sgd_nesterov}]
                [--schedule {polynomial,cyclic}] [--momentum MOMENTUM]
                [-l LEARNING_RATE] [--max-lr MAX_LR] [--min-lr MIN_LR]
                [--power POWER] [--cycle CYCLE]
                               
Start training a semantic segmentation model
optional arguments:

  -h, --help            show this help message and exit

  -m {unet,bayes_segnet,deeplabv3+,fastscnn,separable_unet}, --model {unet,bayes_segnet,deeplabv3+,fastscnn,separable_unet}

                        Specify the model you wish to use: OPTIONS: unet,

                        separable_unet, bayes_segnet, deeplabv3+, fastscnn

  -r RESUME, --resume RESUME

                        Resume the training, specify the weights file path of

                        the format weights-[epoch]-[val-acc]-[val-

                        loss]-[datetime].hdf5

  -p PATH, --path PATH  Specify the root folder for cityscapes dataset, if not
                        used looks for CITYSCAPES_DATASET environment variable
  -c, --coarse          Use the coarse images

  -t TARGET_SIZE, --target-size TARGET_SIZE

                        Set the image size for training, should be a elements

                        of a tuple x,y,c

  -b BATCH, --batch BATCH

                        Set the batch size

  -e EPOCHS, --epochs EPOCHS

                        Set the number of Epochs

  --mixed-precision     Use Mixed Precision. WARNING: May cause memory leaks

  -f {adadelta,adagrad,adam,adamax,ftrl,nadam,rmsprop,sgd,sgd_nesterov}, --lrfinder 
{adadelta,adagrad,adam,adamax,ftrl,nadam,rmsprop,sgd,sgd_nesterov}

                        Use the Learning Rate Finder on a model to determine

                        the best learning rate range for said optimizer

  --schedule {polynomial,cyclic}

                        Set a Learning Rate Schedule, here either Polynomial

                        Decay (polynomial) or Cyclic Learning Rate (cyclic) is

                        Available

  --momentum MOMENTUM   Only useful for lrfinder, adjusts momentum of an

                        optimizer, if there is that option

  -l LEARNING_RATE, --learning-rate LEARNING_RATE

                        Set the learning rate

schedule:

  Arguments for the Learning Rate Scheduler

  --max-lr MAX_LR       The maximum learning rate during training

  --min-lr MIN_LR       The minimum learning rate during training

  --power POWER         The power used in Polynomial Decay

  --cycle CYCLE         The length of cycle used for Cyclic Learning Rates
