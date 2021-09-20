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

train_labels = []
for label_dir in tqdm(sorted(glob.glob("hand_labels/train/label/*.json")), total=1912):
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
    train_labels.append(lbl)
    # lbl = lbl.tolist()
train_labels = np.array(train_labels)
train_labels = train_labels.reshape(1912, -1)
#extra
#train_labels = train_labels.astype(np.float32)
#train_labels = train_labels / 256 - 0.5

test_labels = []
for label_dir in tqdm(sorted(glob.glob("hand_labels/test/label/*.json")), total=846):
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
    test_labels.append(lbl)
    # lbl = lbl.tolist()
test_labels = np.array(test_labels)
test_labels = test_labels.reshape(846, -1)
#extra
#test_labels = test_labels.astype(np.float32)
#test_labels = test_labels / 256 - 0.5  # scale

size_x = 256
size_y = 256
num_classes = 42

# training list

train_images = []
for img_path in tqdm(sorted(glob.glob("hand_labels/train/crop/*.jpg")), total=1912):
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (size_x, size_y))
    train_images.append(img)
"convert list to np array for ml processing"
train_images = np.array(train_images)  # dtype:uint8
#extra
#train_images = train_images.astype(np.float32)
train_images = train_images/255

test_images = []
for img_path in tqdm(sorted(glob.glob("hand_labels/test/crop/*.jpg")), total=846):
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (size_x, size_y))
    test_images.append(img)
"convert list to np array for ml processing"
test_images = np.array(test_images)  # dtype:uint8
#extra
#test_images = test_images.astype(np.float32)
test_images = test_images/255

print('Training image data: ' + str(train_images.shape))
print('Training points data: ' + str(train_labels.shape))
print('Testing image data: ' + str(test_images.shape))
print('Testing points data: ' + str(test_labels.shape))

# plots keypoints on face image
def plot_keypoints(img, points):
    # display image
    plt.imshow(img, cmap='gray')
    #plt.imshow(np.float32(img), cmap='gray')
    # plot the keypoints
    for i in range(0, 42, 2):
        #plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
        plt.scatter(points[i], points[i + 1], color='red')
        # cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
    plt.show()


import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

def rotate_aug(im_tr, pt_tr):
    seq = iaa.Sequential([iaa.Affine(rotate=30, scale=(0.8, 1.2))])
    aug_ims = []
    aug_pts = []
    coordlist = []
    for im, pt in zip(im_tr, pt_tr):
        xcoord = pt[0::2]
        ycoord = pt[1::2]
        for i in range(len(xcoord)):
            coordlist.append(Keypoint(xcoord[i], ycoord[i]))
        kps = KeypointsOnImage(coordlist, shape=im.shape)
        f_im, f_kp = seq(image=im, keypoints=kps)
        all_coords = []
        for k in range(len(kps.keypoints)):
            before = kps.keypoints[k]
            after = f_kp.keypoints[k]
            # print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
            #     i, before.x, before.y, after.x, after.y)
            # )
            all_coords.append(after.x)
            all_coords.append(after.y)
            all_coords_arr = np.asarray(all_coords)
        aug_ims.append(im)
        aug_ims.append(f_im)
        aug_pts.append(pt)
        aug_pts.append(all_coords)
        coordlist.clear()
    return np.asarray(aug_ims), np.asarray(aug_pts)

train_images_aug, train_labels_aug = rotate_aug(train_images, train_labels)

def flip_im_points0(img, points):
  flip_im = np.fliplr(img)
  xcoords = points[0::2]
  ycoords = points[1::2]
  new_points = []
  for i in range(len(xcoords)):
    xp = xcoords[i]
    yp = ycoords[i]
    new_points.append(256-xp)
    new_points.append(yp)
  return flip_im, np.asarray(new_points)

def aug_flip0(im_tr, pt_tr):
  aug_ims = []
  aug_pts = []
  for im, pt in zip(im_tr, pt_tr):
    f_im, f_pts = flip_im_points0(im, pt)
    aug_ims.append(im)
    aug_ims.append(f_im)
    aug_pts.append(pt)
    aug_pts.append(f_pts)
  return np.asarray(aug_ims), np.asarray(aug_pts)

train_images_aug, train_labels_aug = aug_flip0(train_images_aug, train_labels_aug)

print('Training image data: ' + str(train_images_aug.shape))
print('Training points data: ' + str(train_labels_aug.shape))
print('Testing image data: ' + str(test_images.shape))
print('Testing points data: ' + str(test_labels.shape))


id = 50
#plot_keypoints(train_images_aug[id], train_labels_aug[id])
plot_keypoints(train_images[id], train_labels[id])
"""
pickle_out = open("x_train_aug2.pickle","wb")
pickle.dump(train_images_aug, pickle_out)
pickle_out.close()
pickle_out = open("y_train_aug2.pickle","wb")
pickle.dump(train_labels_aug, pickle_out)
pickle_out.close()

pickle_out = open("x_test.pickle","wb")
pickle.dump(test_images, pickle_out)
pickle_out.close()
pickle_out = open("y_test.pickle","wb")
pickle.dump(test_labels, pickle_out)
pickle_out.close()
"""
np.save('xtrain.npy', train_images_aug)
np.save('ytrain.npy', train_labels_aug)
np.save('xtest.npy', test_images)
np.save('ytest.npy', test_labels)
a=1