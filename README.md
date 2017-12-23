# Find Phone

Find the normalized coordinates of a phone in a given image

## Required Python Libraries (non-built-in)

* Numpy
* OpenCV2
* Matplotlib
* roipoly (in the repo)

## Algorithm Overview

Given an image, classify pixels into two classes, "Phone" and "Non-phone", using two Gaussian Classifiers. Collect the classified pixels into a binary mask. Find contours of the binary mask using OpenCV and fit minimum area rectangle on these contours. Find the rectangle with maximum area within a set range, while having a reasonable height and width ratio as an iphone (~1.77). After selecting the best rectangle, return its center point as the predicted coordinates of the phone in the image.  

## Training

First, the training images need to be labeled with the ROI (Region of Interest), which requires the module roipoly. In this algorithm, ROI is the phone screen. The ROI needs to be hand-labeled by a polygon using your mouse cursor. Run the following script:

```
python2 label_images.py ./find_phone
``` 

The command line argument is the path to a folder with labeled images and labels.txt. This script is currently looping through each image with a step size of 4. The step size can be adjusted in the script (same changes must also be made in the training script train_phone_finder.py​, otherwise the mask and image will misalign). After labeling the ROI in all the images, the script will output a file named **phone_screen_masks.npy**. This file will be used in the training script below. Run the following training script:


```
python​ train_phone_finder.py​ ./find_phone
```

The command line argument is the path to a folder with labeled images and labels.txt. This will collect the labeled pixels to compute the mean and covariance parameters for the two Gaussian Classifiers. For convenience, these four parameters are hardcoded in the script. Using these parameters, the script will run the algorithm, predict the coordinates for all the training images, and output the accuracy based on the 0.05 radius range. This script will take a while to run since it runs the algorithm on all training images. The script will also export these four parameters through the file **trained_gaussian_params.npz**, which will be loaded and used by the testing script.

## Testing

The following test script will run the algorithm on the given image and print the predicted coordinates:

```
>> python​ find_phone.py​ ~/find_phone/51.jpg
0.2400 0.6035
```

The command line argument is the path to jpeg image to be tested.

## Utility Module

The module find_phone_utils.py contains functions that are used by both the testing and training script. The core functions are:

* predict_phone_coordinates

## Current State of the Algorithm

The current accuracy on the entire training set is **83.721%**. Since the algorithm is heavily based on the classifier on the pixel values, it is vulnerable to situations where the phone screen has glares/reflections and where the image background has many black pixels that do not belong to the phone. 

The algorithm can certainly be optimized further by picking better parameters for the Gaussian Classifiers, better threshold values for the area of the rectangle and height/width ratio. 