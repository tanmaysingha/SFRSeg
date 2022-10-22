# SFRSeg: Shared Feature Reuse Segmentation Model for Resource Constrained Devices
This is an official site for SFRSeg model. Currently, the model predictions and supplimentary materials are uploaded. Upon the acceptance of the paper, this repository will be updated.

## Datasets
For this research work, we have used Cityscapes, KITTI, CamVid and Indoor objects datasets.
* Cityscapes - To access this benchmark, user needs an account. For test set evaluation, user needs to upload all the test set results into the server. https://www.cityscapes-dataset.com/downloads/ 
* KITTI - To access this benchmark, user needs an account. Like Cityscapes, user needs to submit the test set result to the evaluation server.  http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015    
* CamVid - To access this benchmark, visit this link: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
* Indoor Objects - To access this benchmark, visit this link: https://data.mendeley.com/datasets/hs5w7xfzdk/3

## Class mapping
Different datasets provide different class annotations. For instance, Camvid dataset has 32 class labels. Refer this link to know about all 32 classes of Camvid: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/#ClassLabels. However, literature have shown that all the existing models are trained by 11 classes (Sky, Building, Pole, Road, Sidewalk, Tree, TrafficLight, Fence, Car, Pedestrian, Bicyclist) of Camvid dataset. Thereby, first 32 class annotations of Camvid are converted into 11 class annotations and then model is trained with 11 class annotations. To improve model performance, we also converted Cityscapes 19 class annotations to 11 class anotation and trained the model first with Cityscapes 11 class annotation, then use the pre-trained weight of Cityscapes to train the model with Camvid 11 class annotations. The following table shows the convertion of 32 classes of Camvid dataset to 11 classes.

TrainId | Camvid 11 classes  | Camvid 32 classes   
--------|--------------------|-------------------
   0    |        Sky         | Sky
   1    |     Building       | Archway, Bridge, Building, Tunnel, Wall
   2    |    Column_Pole     | Column_Pole, Traffic Cone
   3    |        Road        | Road, LaneMkgsDriv, LaneMkgsNonDriv  
   4    |      Sidewalk      | Sidewalk, ParkingBlock, RoadShoulder 
   5    |        Tree        | Tree, VegetationMisc
   6    |   TrafficLight     | TrafficLight, Misc_Text, SignSymbol  
   7    |       Fence        | Fence
   8    |        Car         | Car, OtherMoving, SUVPickupTruck, Train, Truck_Bus 
   9    |     Pedestrian     | Animal, CartLuggagePram, Child, Pedestrain   
  10    |     Bicyclist      | Bicyclist, MotorcycleScooter
  
  Note: Void class is not included in the set of 11 classes.
  
  The following table shows the mapping of Cityscapes 19 classes to Camvid 11 classes.
  
TrainId | Camvid 11 classes  | Cityscapes classes   
--------|--------------------|-------------------
   0    |        Sky         | Sky
   1    |     Building       | Building, Wall
   2    |    Column_Pole     | Pole, Polegroup
   3    |        Road        | Road  
   4    |      Sidewalk      | Sidewalk 
   5    |        Tree        | Vegetation
   6    |   TrafficLight     | Traffic Light, Traffic Sign  
   7    |       Fence        | Fence
   8    |        Car         | Car, Truck, Bus, Caravan 
   9    |     Pedestrian     | Person   
  10    |     Bicyclist      | Rider, Bicycle, MotorCycle


## Metrics
To understand the metrics used for model performance evaluation, please  refer here: https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results

## Results
We trained our model by the above mentioned benchmarks at different input resolutions. Cityscapes provides 1024 * 2048 px resolution images. We mainly focus full resolution of cityscapes images. For CamVid dataset, we use 640 * 896 px resolution altough original image size is 720 * 960 px. Similarly, we use 384 * 1280 px resolution input images for KITTI dataset although original size of input image is 375 * 1280 px. For Cityscapes and KITTI datasets, we use 19 classes, however for Camvid dataset we trained the model with 11 classes (suggested by the literature). 

Dataset         | No. of classes  |  Test mIoU | No. of parameters | FLOPs   
----------------|-----------------|------------|-------------------|--------
Cityscapes      |        19       |    70.6%   |    1.6 million    | 37.9 G
KITTI           |        19       |    49.3%   |    1.6 million    |  8.9 G
Camvid          |        11       |    74.7%   |    1.6 million    | 10.2 G
Indoor Objects  |         9       |    60.7%   |    1.6 million    | 19.3 G

Cityscapes, KITTI, CamVid are urban street scenes datasets. Hence, the first three rows in the above table shows the model's performance on outdoor scenes. However, for indoor scenes analysis and indoor navigation mainly for wheelchair users and service robots, we also trained the model with Indoor objects dataset at 768 * 1408 px resolution. FLOPs count varies due to the varied input resolutions.

## FPS (Frame Per Second) count
Model FPS count not only depends on model size, it also depends on input resolution, type of model and most importantly on hardware configuration. Hence, We reproduce some of the existing models based on the literature and train the model under the same system configuration for 10 epochs. We use tensorflow 2.1.0 and keras 2.3.1 for reproducing the existing models. Here is the Google [Colab]https://colab.research.google.com/drive/18kt7LiPDNtjG__VJoAhJBP22C-onD1HY?usp=sharing) link for FPS measurement. You can get weights of different existing models under the weights directory. The size of the weight files of some existing models such as STDC1 and STDC2 is large. You can get these weights from ths following [link](https://drive.google.com/drive/folders/1fwRA-d_A2cYk_H8CzaTigUbp6B4W6rJl?usp=sharing).  


### Cityscapes test results
The output of the test set is submitted to Cityscapes evaluation server. To view the test set result evaluated by the server, click the following link: 
This is an anonymous link given by the Cityscapes server. Upon the acceptance of the paper, test result will be cited by the paper and will be published in the evaluation server.


### Color map of Cityscapes dataset and model prediction using validation sample
![cityscapes_val_set](https://github.com/tanmaysingha/SFRSeg/blob/main/Images/cityscapes_color_map.png?raw=true)
 
### SFRSeg prediction on Cityscapes test samples
![Cityscapes_test_set](https://github.com/tanmaysingha/SFRSeg/blob/main/Images/cityscapes_test_results.png?raw=true)  

### Color map of CamVid dataset and model prediction using validation sample
![CamVid_val_set](https://github.com/tanmaysingha/SFRSeg/blob/main/Images/camvid_color_map.png?raw=true)

### SFRSeg prediction on CamVid validation sample
![CamVid_val_set](https://github.com/tanmaysingha/SFRSeg/blob/main/Images/camvid_test_results.png?raw=true)

### Color map of KITTI dataset and model prediction using validation sample
![CamVid_val_set](https://github.com/tanmaysingha/SFRSeg/blob/main/Images/KITTI_color_map.png?raw=true)

### SFRSeg prediction on KITTI test samples
![KITTI_test_set](https://github.com/tanmaysingha/SFRSeg/blob/main/Images/KITTI_test_results.png?raw=true)

### KITTI test set results
Like Cityscapes, KITTI test set result is also sumbitted to the evaluation server. Click the following link to see the result:
https://github.com/tanmaysingha/SFRSeg/blob/main/Supplementary/KITTI_Test_Results.pdf

### SFRSeg prediction on Indoor objects scenes
![indoor_val_set](https://github.com/tanmaysingha/SFRSeg/blob/main/Images/Indoor_predictions.png?raw=true)
