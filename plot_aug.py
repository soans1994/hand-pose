"""
Process CMU Hand dataset to get cropped hand datasets.
"""

import json
from tqdm import tqdm
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
#train_images = np.load('xtrain.npy', mmap_mode='r')
train_images = np.load('xtrain.npy')
train_labels = np.load('ytrain.npy')
test_images = np.load('xtest.npy')
test_labels = np.load('ytest.npy')

print('Training image data: ' + str(train_images.shape))
print('Training points data: ' + str(train_labels.shape))
print('Testing image data: ' + str(test_images.shape))
print('Testing points data: ' + str(test_labels.shape))

def plot_keypoints(img, points):
    # display image
    #plt.imshow(img, cmap='gray')
    plt.imshow(img)
    #plt.imshow(np.float32(img), cmap='gray')
    # plot the keypoints
    for i in range(0, 42, 2):
        #plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
        plt.scatter(points[i], points[i + 1], color='red', s=1)
        #cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imshow("img",img)
    #cv2.imwrite("img.jpg", img)
    plt.show()

# plots keypoints on face image
def plot_keypoints1(img, points):

    for i in range(0, 42, 2):
        cv2.circle(img, (int(points[i]), int(points[i + 1])), 5, (0, 0, 255), thickness=-1)  # , lineType=-1)#, shift=0)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("1.jpg", img)
    #plt.show()
# plots keypoints on face image
def plot_keypoints2(img, points):
    for i in range(0, 42, 2):
        cv2.circle(img, (int(points[i]), int(points[i + 1])), 5, (0, 0, 255), thickness=-1)  # , lineType=-1)#, shift=0)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("2.jpg", img)
    #plt.show()
# plots keypoints on face image
def plot_keypoints3(img, points):
    for i in range(0, 42, 2):
        cv2.circle(img, (int(points[i]), int(points[i + 1])), 5, (0, 0, 255), thickness=-1)  # , lineType=-1)#, shift=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("3.jpg", img)
    #plt.show()
def plot_keypoints4(img, points):
    for i in range(0, 42, 2):
        cv2.circle(img, (int(points[i]), int(points[i + 1])), 5, (0, 0, 255), thickness=-1)  # , lineType=-1)#, shift=0)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("4.jpg", img)
    #plt.show()

train_images=train_images*255
#plot_keypoints(train_images_aug[id], train_labels_aug[id])
#plot_keypoints(train_images[id], train_labels[id])
#plot_keypoints(train_images[id], train_labels[id])
id = 0
plot_keypoints1(train_images[id], train_labels[id])
plot_keypoints2(train_images[id+1], train_labels[id+1])
plot_keypoints3(train_images[id+2], train_labels[id+2])
plot_keypoints4(train_images[id+3], train_labels[id+3])
a=1