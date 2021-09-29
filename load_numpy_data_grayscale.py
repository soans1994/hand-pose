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
    img = cv2.imread(img_path,0)
    # img = cv2.resize(img, (size_x, size_y))
    train_images.append(img)
"convert list to np array for ml processing"
train_images = np.array(train_images)  # dtype:uint8
#extra
train_images = train_images.astype(np.float32)
train_images = train_images/255
train_images = train_images.reshape(train_images.shape[0], 256, 256, 1)

test_images = []
for img_path in tqdm(sorted(glob.glob("hand_labels/test/crop/*.jpg")), total=846):
    img = cv2.imread(img_path,0)
    # img = cv2.resize(img, (size_x, size_y))
    test_images.append(img)
"convert list to np array for ml processing"
test_images = np.array(test_images)  # dtype:uint8
#extra
test_images = test_images.astype(np.float32)
test_images = test_images/255
test_images = test_images.reshape(test_images.shape[0], 256, 256, 1)

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
#"""
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
#"""
"""
flip_img, flip_points = augment_data(train_images[0], train_labels[0])
plot_keypoints(flip_img, flip_points)
flip_img, flip_points = augment_data(train_images[1], train_labels[1])
plot_keypoints(flip_img, flip_points)
flip_img, flip_points = augment_data(train_images[19], train_labels[19])
plot_keypoints(flip_img, flip_points)
flip_img, flip_points = augment_data(train_images[20], train_labels[20])
plot_keypoints(flip_img, flip_points)
flip_img, flip_points = augment_data(train_images[50], train_labels[50])
plot_keypoints(flip_img, flip_points)
flip_img, flip_points = augment_data(train_images[60], train_labels[60])
plot_keypoints(flip_img, flip_points)
#plot_keypoints(train_images[0], train_labels[0])
#plot_keypoints(train_images[1], train_labels[1])
#plot_keypoints(train_images[19], train_labels[19])
#plot_keypoints(train_images[20], train_labels[20])
plot_keypoints(train_images[60], train_labels[60])
plt.show()
"""
train_images_aug = []
train_labels_aug = []

#"""
# apply flipping operation
for i in tqdm(range(0, train_images.shape[0])):
    aug_img, aug_point = augment_data(train_images[i], train_labels[i])
    # original data
    train_images_aug.append(train_images[i])
    train_labels_aug.append(train_labels[i])

    # augmented data
    train_images_aug.append(aug_img)
    train_labels_aug.append(aug_point)
"""
# convert to numpy
"""
train_images_aug = np.array(train_images_aug)
train_labels_aug = np.copy(train_labels_aug)
"""
"""
print('Training image data: ' + str(train_images_aug.shape))
print('Training points data: ' + str(train_labels_aug.shape))
print('Testing image data: ' + str(test_images.shape))
print('Testing points data: ' + str(test_labels.shape))
"""
id = 50
#plot_keypoints(train_images_aug[id], train_labels_aug[id])
plot_keypoints(train_images[id], train_labels[id])
"""
fig = plt.figure(figsize=(20,20))
for i in range(16):
    fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_keypoints(train_images_aug[i], train_labels_aug[i])
plt.show()

# train_images = train_images.reshape(train_images.shape[0], 256, 256, 1)
#train_images2 = train_images.reshape(train_images.shape[0], 256, 256, 3)
#img_height = train_images2.shape[1]
#img_width = train_images2.shape[2]
#img_channels = train_images2.shape[3]
img_height = 256
img_width = 256
img_channels = 1
input_shape = (img_height, img_width, img_channels)
print(input_shape)

pickle_out = open("x_train.pickle2","wb")
pickle.dump(train_images_aug, pickle_out)
pickle_out.close()
pickle_out = open("y_train.pickle2","wb")
pickle.dump(train_labels_aug, pickle_out)
pickle_out.close()

pickle_out = open("x_test.pickle2","wb")
pickle.dump(test_images, pickle_out)
pickle_out.close()
pickle_out = open("y_test.pickle2","wb")
pickle.dump(test_labels, pickle_out)
pickle_out.close()


a=1