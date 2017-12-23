import cv2
import matplotlib.pylab as pl
import numpy as np
import os
from roipoly import roipoly
import sys


masks = []

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

# training with every 4th image
image_step_size = 4
            
for i in range(0, len(lines), image_step_size):  # looping through images with a step size
# for i in range(len(lines)):  # using every image 
    
    l = lines[i]
    print "File name: " + str(l[0])
    
    # read the image and mark the red regions with polygons
    img = cv2.imread(os.path.join(path, l[0]))
    pl.imshow(img)
    MyROI = roipoly(roicolor='r') #let user draw ROI 
    masks.append(MyROI.getMask(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)))

# save the masks with numpy into .npy for the training code
np.save("phone_screen_masks.npy",masks)
