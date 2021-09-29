"""
Process CMU Hand dataset to get cropped hand datasets.
"""
import keras
import json
from tqdm import tqdm
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint
import imgaug.augmenters as iaa

IMG_SIZE = 256
NUM_KEYPOINTS = 42

class generator(keras.utils.Sequence):
    def __init__(self, image_keys, aug, batch_size, train=True):
        self.image_keys = image_keys
        self.aug = aug
        self.batch_size = batch_size
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_keys) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_keys))
        if self.train:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        image_keys_temp = [self.image_keys[k] for k in indexes]
        images = self.__data_generation(image_keys_temp)

        return images

    def __data_generation(self, image_keys_temp):
        batch_images = np.empty((self.batch_size, IMG_SIZE, IMG_SIZE, 3), dtype="int")
        batch_keypoints = np.empty((self.batch_size, NUM_KEYPOINTS), dtype="float32")

        for i, key in enumerate(image_keys_temp):
            data = cv2.imread(key)
            split = os.path.split(key)
            #print(split)
            #print(split[1])
            extension = os.path.splitext(split[1])[0]
            #print(extension)
            if self.image_keys == samples2:
                key2 = "hand_labels/test/label/" + extension + ".json"
            else:
                key2 = "hand_labels/train/label/" + extension + ".json"
            #print(key2)
            #data = cv2.resize(data, (256, 256))
            # We then project the original image and its keypoint coordinates.
            #current_image = data
            # Apply the augmentation pipeline.
            #new_image = self.aug(image=current_image)
            #new_image = current_image
            #batch_images[i,] = new_image
            batch_images[i,] = data
            dat = json.load(open(key2))
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

            pts = pts[:, :2] - hand_box[0, :]
            # current_keypoint = np.array(data["joints"])[:, :2]
            # kps = []
            pts = pts * 256 / width
            # More on why this reshaping later.
            # batch_keypoints[i,] = np.array(kp_temp).reshape(1, 1, 42)#same as below
            batch_keypoints[i,] = np.array(pts).reshape(-1, 42)  # same as above
            # Scale the coordinates to [0, 1] range.
        return batch_images, batch_keypoints

def plot_keypoints(img, points):
    # display image
    plt.imshow(img, cmap='gray')
    #plt.imshow(np.float32(img), cmap='gray')
    # plot the keypoints
    for i in range(0, 42, 2):
        #plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
        plt.scatter(points[i], points[i + 1], color='red')
        #plt.scatter(points[:, 0], points[:, 1])
        #cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
    plt.show()

samples = sorted(glob.glob("hand_labels/train/crop/*.jpg"))
samples2 = sorted(glob.glob("hand_labels/test/crop/*.jpg"))
x = generator(samples, batch_size=32, aug=None)#, aug=train_aug)
y = generator(samples2, batch_size=32, aug=None)#, aug=train_aug)
#train_images, train_labels = generator(samples, batch_size=32, aug=None)#, aug=train_aug)
#print(len(x), len(y))
for i,j in x:
    print(i.shape, j.shape)
    print(i[0].shape, j[0].shape)
    break
plot_keypoints(i[0], j[0])
for i,j in y:
    print(i.shape, j.shape)
    print(i[0].shape, j[0].shape)
    break
plot_keypoints(i[0], j[0])
plt.show()