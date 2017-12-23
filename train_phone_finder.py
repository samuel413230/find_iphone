import cv2
import math
import numpy as np
from numpy.linalg import inv, det
import os
import sys
from find_phone_utils import predict_phone_coordinates


## Training: Find the parameters for the Gaussian Classifier on pixels ##

# read command line argument
assert len(sys.argv) == 2, 'Only 1 command line arg: path to images and labels.txt'

path = sys.argv[1]
filename = 'labels.txt'

with open(os.path.join(path, filename), 'r') as f:
    lines = f.readlines()  # read lines into a list
    
    assert len(lines) != 0, 'labels.txt is empty'
    
    # strip '\n', split each line by whitespace 
    lines = [l.strip('\n').split(' ') for l in lines] 

    for line in lines:
        assert len(line) == 3, \
            'A line in labels.txt does not contain 3 entries: %s' % line

# load in hand-labeled masks (made using roipoly) for each image
masks = np.load("phone_screen_masks.npy")

coord = []  # store the labeled coordinates for the training images
train_images = []
phone_px = []  # store the pixels belong to class "Phone"
non_phone_px = []  # store the pixels belong to class "Non-phone"


# training with every 4th image
image_step_size = 4

for i in range(0, len(lines), image_step_size): 
    
    l = lines[i]
    coord.append([l[1], l[2]])
    
    img = cv2.imread(os.path.join(path, l[0]))
    
    mask = masks[i/image_step_size]
    
    # FOR DEBUGGING: to make sure image and mask is aligned correctly
#     print "File name: " + str(l[0])
#     plt.imshow(img)
#     plt.show()
#     plt.imshow(mask)
#     plt.show()
    
    # convert image to HSV from RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # append image to the list of images
    train_images.append(img)
    
    # flaten the image and mask
    img = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    mask = np.reshape(mask, mask.shape[0] * mask.shape[1])
    
    # sort the phone and non-phone pixels based on the mask
    for j in range(len(mask)):
        if mask[j] == True:
            phone_px.append(img[j])
        else:
            non_phone_px.append(img[j])

# convert list to np array
coord = np.array(coord)
train_images = np.array(train_images)
phone_px = np.array(phone_px)
non_phone_px = np.array(non_phone_px)

    
# calculate mean and cov for each Gaussian Classifer
# two Gaussian Classifier: Phone vs. Non-phone
phone_px_mean = np.mean(phone_px,axis=0)
phone_px_cov = np.cov(phone_px.T)

non_phone_px_mean = np.mean(non_phone_px,axis=0)
non_phone_px_cov = np.cov(non_phone_px.T)


## Parameters found during my own training with every other 4 image ##

phone_px_mean = np.array([ 112.89214859,  108.55965949,   37.51568876])

phone_px_cov = np.array([[ 1071.16904691,   397.4972179 ,   115.48311763],
                         [  397.4972179 ,  3302.48806788,  -467.21693549],
                         [  115.48311763,  -467.21693549,  1020.85471058]])

non_phone_px_mean = np.array([  49.11371059,   17.46024866,  149.36317865])

non_phone_px_cov = np.array([[ 3367.83691787,  -286.18052514,   317.55961653],
                             [ -286.18052514,   418.24596306,  -235.06705808],
                             [  317.55961653,  -235.06705808,  1292.63417352]])

# save these four parameters for find_phone.py
np.savez("trained_gaussian_params.npz", phone_px_mean=phone_px_mean, phone_px_cov=phone_px_cov, 
         non_phone_px_mean=non_phone_px_mean, non_phone_px_cov=non_phone_px_cov)


# let's run the algorithm on all train images and calculate accuracy

coord = []  # store the labeled coordinates for the training images
train_images = []

# load in all training images
for i in range(len(lines)):
    l = lines[i]
    coord.append([l[1], l[2]])  # store the labeled coordinates 
    
    # read the image and mark the red regions with polygons
    img = cv2.imread(os.path.join(path, l[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to HSV
    
    # append image to the list of images to be tested
    train_images.append(img)
    
# convert list to np array
coord = np.array(coord)
train_images = np.array(train_images)

count = 0 # for keeping track of accuracy

for i in range(train_images.shape[0]):
    img = train_images[i]
    
    pred_coord = predict_phone_coordinates(img, phone_px_mean, phone_px_cov, 
                                       non_phone_px_mean, non_phone_px_cov)
    
    # determine if output is correct within 0.05 normalized distance
    distance = np.linalg.norm(pred_coord - coord[i].astype(dtype='float'))
    
    if distance <= 0.05:
        count += 1
    
    # To print out details about each training image result:
#     import matplotlib.pyplot as plt
#     print "Box Area: " + str(box_area_max)
#     print "Actual coord: " + str(coord[i])
#     print "Box coord: " + str(pred_coord)
#     print "Error: " + str(np.linalg.norm(pred_coord - coord[i].astype(dtype='float')))
#     barrel_pts_float = cv2.boxPoints(box_max)
#     barrel_pts = np.int0(barrel_pts_float)

#     barrel_img = np.zeros((IMG_H,IMG_W))
#     cv2.drawContours(barrel_img,[barrel_pts],0,(255,0,0),2)
#     plt.imshow(barrel_img)
#     plt.show()

print "Accuracy: {0}%".format(str(count*100.0/train_images.shape[0]))