import lesson_functions
import os
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from skimage import color, exposure
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import time
from IPython.display import HTML
import collections

####################################################################################################################
#---------------------------------------- Capturing Video Frames------------------------------------------------------------
####################################################################################################################

#video_in = 'project_video.mp4'
#vidcap = cv2.VideoCapture(video_in)
#
#success,image = vidcap.read()
#count = 0;
#while success:
#    success,image = vidcap.read()# image is an array of array of [R,G,B] values
#    cv2.imwrite("output_images/frame%d.png" % count, image)     # save frame as JPEG file
#    if cv2.waitKey(10) == 27:                     # exit if Escape is hit
#        break
#    count += 1
#print("frames saved.")

####################################################################################################################
#---------------------------------------- Importing Data------------------------------------------------------------
####################################################################################################################
video_frames = glob.glob('output_images/*.png')
# print(len(video_frames))
i = 0
for img in video_frames:
    image = cv2.imread(img)
    #     cv2.imwrite("output_images/retyped%d.png" % i, image)
    image = cv2.resize(image, (64,64))
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("output_images/beginingShadows%d.png" % i, image)
    i += 1
print("images resized.")

#white_cars = glob.glob('WhiteCar/*.jpg')
## print(len(video_frames))
#i = 0
#for img in white_cars:
#    image = cv2.imread(img)
#    cv2.imwrite("WhiteCar/retyped%d.png" % i, image)
#    image = cv2.resize(image, (64,64))
##    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    cv2.imwrite("WhiteCar/resizedframe%d.png" % i, image)
#    i += 1
#print("images resized.")
