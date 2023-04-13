# TSA Prohibited Object Detection
## Introduction
In this project, we propose a research about using deep learning and machine learning learning methods to perform detection of prohibited items from X-Ray images. In this study, we utilize two object detection methods: Faster R-CNN, and Histogram of Oriented Gradients (HOG) with Support Vector Machine (SVM). With a security dataset PIDray, we build, train, evaluate, and compare these methods. We expect to contribute to improve the performance of security detection. Specifically, the contribuions of this project are:

-   Propose a system that use Faster R-CNN to detect prohibited items from X-Ray images. We use pre-trained Faster R-CNN model and not pre-trained Faster R-CNN model to build the system and compare their performance.

-   Propose a system that use HOG and SVM to perform prohibited item detection from X-Ray images. The HOG features are displayed by using HOGgles

-   Compare these methods' detection performance base on multiple factors.

## Dataset
For this project, we use PIDray dataset. This dataset is in COCO format, and contains 47,677 slices of X-Ray images collected in different scenarios (such as airport, subway stations, and railway stations). It covers total 12 categories of prohibited items, namely guns, knife, wrench, pliers, scissors, hammer, handcuffs, baton, sprayer, power-bank, lighter and bullet. You can find more information about this dataset at [here](https://github.com/bywang2018/security-dataset).

## Methods
### Faster R-CNN
For Faster R-CNN, we use MMDetection package to build, train, and evaluate the models. You can find more information about this package from [here](https://mmdetection.readthedocs.io/en/latest/). 

We conduct two experiments based on this model architecture. The first experiment is to train the model from scratch, which means the weights are randomly initialized. The second experiment focuses on the behavior of the pretrained model. We load the model with weights pretrained on the COCO dataset.

### HOG
For this experiment, we scale the image to $320 \times 320$, and normalize them with 0.5 for mean's value and 0.5 for standard deviation value. Then we iterate through the dataloader and compute HOG features for each image by using scikit-image. After getting the HOG featurs for each images, we train the SVM classifier with scikit-learn 

## Usage
Please install mmdetection package first. You can follow the installation steps from this [tutorial](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)

For Faster R-CNN, please run the following commands to train the model
```
$ cd Faster_R_CNN_Obj_Detect
$ python train.py config.py
```

For HOG, please run the following commands to train the model
```
$ cd HOG_SVM_Obj_Detect
$ python main.py
```

Please change the address paths in the files when you run the codes.

## Notice
This project is not finished yet. We are still running and testing for HOG. More information will be provided later.