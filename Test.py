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
#---------------------------------------- Importing Data------------------------------------------------------------
####################################################################################################################
print("#------------ Importing Data ---------")
# First We need to read in the images.
# The data base is divided to notcar and car vehicles provided with different sources.

notcar_dir = 'non-vehicles/'
car_dir = 'vehicles/'
notcar_subdir = os.listdir(notcar_dir)
car_subdir = os.listdir(car_dir)

cars = []
not_cars = []

print("dires", notcar_dir, notcar_subdir)
print("dires", car_dir, car_subdir)

# for sub_dirs in notcar_subdir:
#     image = glob.glob(notcar_dir+sub_dirs+'/*')
#     not_cars.extend(image)

image = glob.glob(notcar_dir+notcar_subdir[1]+'/*')
not_cars.extend(image)
image = glob.glob(notcar_dir+notcar_subdir[2]+'/*')
not_cars.extend(image)
image = glob.glob(notcar_dir+notcar_subdir[3]+'/*')
not_cars.extend(image)
image = glob.glob(notcar_dir+notcar_subdir[4]+'/*')
not_cars.extend(image)


# for sub_dirs in car_subdir:
#     image = glob.glob(car_dir+sub_dirs+'/*')
#     cars.extend(image)

image = glob.glob(car_dir+car_subdir[1]+'/*')
cars.extend(image)
image = glob.glob(car_dir+car_subdir[2]+'/*')
cars.extend(image)
image = glob.glob(car_dir+car_subdir[3]+'/*')
cars.extend(image)
image = glob.glob(car_dir+car_subdir[4]+'/*')
cars.extend(image)
image = glob.glob(car_dir+car_subdir[5]+'/*')
cars.extend(image)
image = glob.glob(car_dir+car_subdir[6]+'/*')
cars.extend(image)

print("Number of Vehicles and non-vehicles: ", len(cars),len(not_cars))

with open("cars.txt", 'w') as f:
    for car in cars:
        f.write(car+'\n')

with open("notcars.txt", 'w') as f:
    for notcars in not_cars:
        f.write(notcars+'\n')


cars = np.array(cars)
not_cars = np.array(not_cars)

####################################################################################################################
#---------------------------------------- Training Classifier-------------------------------------------------------
####################################################################################################################
print()
print("#------------ Training Classifier---------")
cars_sample = cars
notcars_sample = not_cars

sample_cars = np.array(cars_sample)
notcars_sample = np.array(notcars_sample)
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations- it was 9
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
car_features = extract_features(sample_cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat )

notcar_features = extract_features(notcars_sample, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

print("notcar_features and car_features length ", len(notcar_features), len(car_features))

#notcar_features = np.array(notcar_features)
#car_features = np.array(car_features)
#print("notcar_features and car_features shape: ", notcar_features.shape, car_features.shape)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
# svc = LinearSVC()
# Set the parameters by cross-validation
svc = GridSearchCV(LinearSVC(), param_grid={'C':np.logspace(-3,-2,5)})
# svc = GridSearchCV(LinearSVC(), param_grid=[{'C': [0.001, 0.01, 1, 10, 100, 1000]}])
# Check the training time for the SVC
print("Training ...")
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()

print(round(t2-t, 2), 'Seconds to train SVC...')
print("GridSearchCV results : ", svc.best_estimator_)
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
####################################################################################################################
#---------------------------------------- Pipeline and Filtering----------------------------------------------------
####################################################################################################################
print()
print("#------------ Testing on Test Images ---------")

y_start_stop = [400, 656] # Min and max in y to search in slide_window()
images = glob.glob('test_images/*.jpg')
scale = 1.5
decision_th = 0.5
threshold = 0.5
heatmaps_collection = collections.deque(maxlen=2)
multiple_detection = []
labeled_images = []
heatmap_array = []
output_images = [] # stored draw_img in this array

### single-scale pipeline ###
for img in images:
    image = mpimg.imread(img)
    out_img, heat = find_cars(image, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, decision_th)

    # These are just for the debuging purposes
    multiple_detection.append(out_img)
    # Apply threshold to help remove false positives
    heat_threshold = apply_threshold(heat,threshold)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat_threshold, 0, 255)
    heatmap_array.append(heatmap)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    labeled_images.append(labels)
    draw_img = draw_labeled_bboxes(np.copy(image), labels) #image
    output_images.append(draw_img)

#fig = plt.figure(figsize=(12,18), dpi=300)
for i, img in enumerate(multiple_detection):
    plt.subplot(5, 2, i+1)
    plt.imshow(img)

plt.show()

# Heatmaps plots
#fig2 = plt.figure(figsize=(12,18), dpi=300)
for i, img in enumerate(heatmap_array):
    plt.subplot(5, 2, i+1)
    plt.imshow(img)
    plt.title('Heat Map')
#    fig2.tight_layout()
plt.show()

# Lables plots
#fig3 = plt.figure(figsize=(12,18), dpi=300)
for i, item in enumerate(labeled_images):
    plt.subplot(5, 2, i+1)
    plt.imshow(item[0], cmap='gray')
    plt.title('Labels')
#    fig3.tight_layout()
plt.show()


# Final results on test images
#fig4 = plt.figure(figsize=(12,18), dpi=300)
for i, img in enumerate(output_images):
    plt.subplot(5, 2, i+1)
    plt.imshow(img)
    plt.title('Car Positions')
#    fig4.tight_layout()



####################################################################################################################
#---------------------------------------- Video Processing ---------------------------------------------------------
####################################################################################################################
print()
print("#------------ Running the video Pipeline ---------")
heatmaps = collections.deque(maxlen=8) #6
threshold = 5.0
decision_th = 0.5
scale = 1.5
# pipeline
def image_pipeline(image):
    out_img, heat = find_cars(image, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, decision_th)
    heatmaps_collection.append(heat)
    heatmap_sum = sum(heatmaps_collection)
    # Apply threshold to help remove false positive
    heat = apply_threshold(heat,threshold) #4
    heatmap = np.clip(heat , 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img


video_in = 'project_video.mp4'
video_out = 'output_video.mp4'

clip = VideoFileClip(video_in)   # remember the output of this function is RGB
video_clip = clip.fl_image(image_pipeline)
video_clip.write_videofile(video_out, audio=False)
