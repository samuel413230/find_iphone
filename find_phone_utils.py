import cv2
import math
import numpy as np
from numpy.linalg import inv, det


def gaussian_d(mean, cov, px):
    """Given the mean, cov, and input vector to compute the likelihood
    
    Args:
        mean: Numpy array, mean of the Gaussian distribution
        cov: Numpy array, covariance of the Gaussian distribution
        px: Numpy array, input vector in the space
        
    Returns:
        A scalar probability value, representing the likelihood of the 
        input vector with the given Gaussian distribution parameters
    """

    exponent = -0.5*(np.dot(np.dot(((px - mean.T)),(inv(cov))),(px - mean)))
    
    denom = (((2*math.pi)**len(mean))*(det(cov)))**(0.5)
    
    return np.exp(exponent) / denom


def predict_phone_coordinates(img, phone_px_mean, phone_px_cov, non_phone_px_mean, non_phone_px_cov):
    """Predict the normalized coordinates of the phone in the given image
    
    Args:
        img: an Numpy ndarray, image in the HSV space (NOT BGR!!)
        phone_px_mean: mean for the pixel class "Phone"
        phone_px_cov: covariance for the pixel class "Phone"
        non_phone_px_mean: mean for the pixel class "Non-Phone"
        non_phone_px_cov: covariance for the pixel class "Non-Phone"
        
    Returns:
        A Numpy array, contains two elements: the predicted normalized coordinates x, y
    
    """
    
    # get height and width of the image
    IMG_H = img.shape[0]
    IMG_W = img.shape[1]

    img = np.reshape(img, (IMG_H*IMG_W, 3))  # flatten the image

    # create a binary mask by classifying the pixels into "Phone" vs. "Non-phone"

    testset_mask = []

    for px in img:
        if gaussian_d(phone_px_mean, phone_px_cov, px) >= gaussian_d(non_phone_px_mean, non_phone_px_cov, px):
            testset_mask.append(True)
        else:
            testset_mask.append(False)

    testset_mask = np.asarray(testset_mask).reshape(IMG_H,IMG_W)

    # find the contours on the mask
    img, contours, hierarchy = cv2.findContours(testset_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    box_area_max = 0
    box_max = 0
    for contour in contours:
        box = cv2.minAreaRect(contour)  # fit bounding rect on contour
        box_w = box[1][0]
        box_h = box[1][1]

        # if width is greater than height, swap them
        if box_w > box_h:
            box_h = box[1][0]
            box_w = box[1][1]

        # ratio between height and width of an iphone 6 is 1.7786
        # define upper limit of height and width ratio
        H_W_LIMIT_U = 1.7786 * 1.75 

        # define max area for classifying an iphone
        AREA_LIMIT = 1400

        if (box_h != 0) and (box_w != 0):  # to prevent error in box_h/box_w
            box_area = box_h*box_w

            # if box area > previous max area and height/width ratio are in the reasonable range
            if (AREA_LIMIT > box_area > box_area_max) and (1.10 < box_h/box_w < H_W_LIMIT_U):
                box_area_max = box_area
                box_max = box

    if box_max:
        pred_coord = np.array([box_max[0][0]/IMG_W, box_max[0][1]/IMG_H])  # predict coordiniates
    else:
        pred_coord = np.array([0.5, 0.5])  # in case of detection failure

    return pred_coord