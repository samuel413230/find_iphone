import cv2
import math
import numpy as np
from numpy.linalg import inv, det
import os
import sys
from find_phone_utils import predict_phone_coordinates


# read command line argument
assert len(sys.argv) == 2, 'Only 1 command line arg: path to the image to be tested'

img_path = sys.argv[1]


# read parameters from train_phone_finder.py for the Gaussian Classifiers
data = np.load("trained_gaussian_params.npz")
data.keys()

phone_px_mean = data['phone_px_mean']
phone_px_cov = data['phone_px_cov']
non_phone_px_mean = data['non_phone_px_mean']
non_phone_px_cov = data['non_phone_px_cov']


# load in test image 
img = cv2.imread(img_path)

# convert image to HSV from RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

pred_coord = predict_phone_coordinates(img, phone_px_mean, phone_px_cov, 
                                       non_phone_px_mean, non_phone_px_cov)

print "%.4f %.4f" % (pred_coord[0] ,pred_coord[1])