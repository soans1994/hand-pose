"""
Process CMU Hand dataset to get cropped hand datasets.
"""
import os
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import glob
import cv2
from matplotlib import pyplot as plt

size_x = 256
size_y = 256

train_labels=[]
for label_dir in tqdm(sorted(glob.glob("hand_labels/train/label/*.json")),total=1912):
   # label_dir = 'hand_labels/test/label/' + img[:-4] + '.json'

    dat = json.load(open(label_dir))
    pts = np.array(dat['hand_pts'])

    lbl = pts[:, :2]
    train_labels.append(lbl)

    #lbl = lbl.tolist()
train_labels = np.array(train_labels)#float64 dont change
train_labels = train_labels.reshape(1912,-1)

train_images = []
for img_path in tqdm(sorted(glob.glob("hand_labels/train/data/*.jpg")), total=1912):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, (size_x, size_y))
    train_images.append(img)
"convert list to np array for ml processing"
train_images = np.array(train_images)  # dtype:uint8


train_labels_crop=[]
for label_dir in tqdm(sorted(glob.glob("hand_labels/train/label/*.json")),total=1912):
   # label_dir = 'hand_labels/test/label/' + img[:-4] + '.json'

    dat = json.load(open(label_dir))
    pts = np.array(dat['hand_pts'])
    xmin = min(pts[:, 0])
    xmax = max(pts[:, 0])
    ymin = min(pts[:, 1])
    ymax = max(pts[:, 1])

    B = max(xmax - xmin, ymax - ymin)
    # B is the maximum dimension of the tightest bounding box
    width = 2.2 * B  # This is based on the paper

    # the center of hand box can be
    center = dat["hand_box_center"]
    hand_box = [[center[0] - width / 2., center[1] - width / 2.],
                [center[0] + width / 2., center[1] + width / 2.]]
    hand_box = np.array(hand_box)

    lbl = pts[:, :2] - hand_box[0, :]
    lbl = lbl * 256 / width
    train_labels_crop.append(lbl)

    #lbl = lbl.tolist()
train_labels_crop = np.array(train_labels_crop)#float64 dont change
train_labels_crop = train_labels_crop.reshape(1912,-1)

train_images_crop = []
for img_path in tqdm(sorted(glob.glob("hand_labels/train/crop/*.jpg")), total=1912):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size_x, size_y))
    train_images_crop.append(img)
"convert list to np array for ml processing"
train_images_crop = np.array(train_images_crop)  # dtype:uint8

# plots keypoints on face image
def plot_keypoints(img, points):
    # display image
    #plt.imshow(img, cmap='gray')
    #plt.imshow(img)
    #plt.imshow(np.float32(img), cmap='gray')
    # plot the keypoints
    for i in range(0, 42, 2):
        #plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
        #plt.scatter(points[i], points[i + 1], color='red', s=1)
        cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imshow("img",img)
    cv2.imwrite("img.jpg", img)
    #plt.show()

# plots keypoints on face image
def plot_keypoints2(img, points):
    # display image
    #plt.imshow(img, cmap='gray')
    #plt.imshow(img)
    #plt.imshow(np.float32(img), cmap='gray')
    # plot the keypoints
    for i in range(0, 42, 2):
        #plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
        #plt.scatter(points[i], points[i + 1], color='red', s=1)
        cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imshow("img",img)
    cv2.imwrite("img_crop.jpg", img)
    #plt.show()

# plots keypoints on face image
def plot_keypoints3(img, points):
    # display image
    #plt.imshow(img, cmap='gray')
    #plt.imshow(img)
    #plt.imshow(np.float32(img), cmap='gray')
    # plot the keypoints
    for i in range(0, 42, 2):
        #plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
        #plt.scatter(points[i], points[i + 1], color='red', s=1)
        cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imshow("img",img)
    cv2.imwrite("img_flip.jpg", img)
    #plt.show()

# plots keypoints on face image
def plot_keypoints4(img, points):
    # display image
    #plt.imshow(img, cmap='gray')
    #plt.imshow(img)
    #plt.imshow(np.float32(img), cmap='gray')
    # plot the keypoints
    for i in range(0, 42, 2):
        #plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
        #plt.scatter(points[i], points[i + 1], color='red', s=1)
        cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imshow("img",img)
    cv2.imwrite("img_rot.jpg", img)
    #plt.show()


# does data augmentation by flipping the image
def augment_data(img, points):
    rows, cols, channel = img.shape
    new_img = np.copy(img)

    # flip the image
    for i in range(256):
        for j in range(128):
            temp = img[i][j]
            new_img[i][j] = img[i][cols - j - 1]
            new_img[i][cols - j - 1] = temp

    # flip the points
    new_points = np.copy(points)
    for i in range(0, 42, 2):
        #new_points[i] = -points[i]
        new_points[i] = 256-points[i]-1
        #new_points[i] = 256 - points[i] # 1 pixel differnec

    return new_img, new_points


id = 500
#train_images_flip, train_labels_flip = augment_data(train_images_crop[id], train_labels_crop[id])
#plot_keypoints(train_images_aug[id], train_labels_aug[id])
plot_keypoints(train_images[id], train_labels[id])
plot_keypoints2(train_images_crop[id], train_labels_crop[id])
a=1