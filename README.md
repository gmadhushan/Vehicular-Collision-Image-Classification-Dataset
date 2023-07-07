# Vehicular-Collision-Image-Classification-Dataset
This repository contains the Vehicular Collision Image Classification Dataset created from corwd-sourced dashboard camera videos containing on-road vehicular collisions.

It is a three-class image classification problem consisting of images belonging to "No Collision", "Collision" and "Collided" classes.
The vehicular collision image classification dataset consists of a total of 8931 images.
The training - validation - test data splits are formed with 70 % - 15 % - 15 % of images from the dataset.
The training and validation data consists of 2600 images for No Collision, 2490 images for Collision and 2500 images for Collided classes respectively.
The test data consists of 390 images for No Collision, 374 images for Collision and 375 images for Collided classes respectively.

**Download Dataset**

The Dataset can be downloaded from the following link upon request.

https://drive.google.com/drive/folders/1VfpBJRSVit3Nu6rJzwFbTH17U3KDwgeg?usp=sharing


**Example Images from the Dataset**

![Github_Sample_1](https://github.com/gmadhushan/Vehicular-Collision-Image-Classification-Dataset/assets/62023065/ec2c20ff-94a8-47a7-a4ac-b9d14ed9c0e2)

/home/madhu-phd/Pictures/Screenshots/Github_Sample_1.png



**Inference**

The best performing image classification deep learning model built on MobileNet backbone using the Teachable Machine on this proposed dataset can be downloaded from the folder "Model".

Inference on the test data can be run using the "Inference_Metrics.py" file.
Inference on individial classes of test data and saving of output images with prediction class and confidence score can be done using the "Inference_ImageOut.py" file.

**Sample Inputs and Results**

![Github_Sample_2](https://github.com/gmadhushan/Vehicular-Collision-Image-Classification-Dataset/assets/62023065/d78c964b-f022-4b9f-84bc-474f3016a6b1)
