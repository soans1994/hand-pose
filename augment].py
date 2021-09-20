
import json
from tqdm import tqdm
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

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


train_images = []
for img_path in tqdm(sorted(glob.glob("hand_labels/train/crop/*.jpg")), total=1912):
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (size_x, size_y))
    train_images.append(img)
"convert list to np array for ml processing"
train_images = np.array(train_images)  # dtype:uint8
#extra
train_images = train_images.astype(np.float32)
train_images = train_images/255


# plots keypoints on face image
def plot_keypoints(img, points):
    # display image
    plt.imshow(img, cmap='gray')
    # plot the keypoints
    for i in range(0,42,2):
        #plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
        plt.scatter(points[i] , points[i + 1], color='red')

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
        new_points[i] = -points[i]

    return new_img, new_points

flip_img, flip_points = augment_data(train_images[0], train_labels[0])
plot_keypoints(flip_img, flip_points)
plt.show()

train_images_aug = []
train_labels_aug = []

# apply flipping operation
for i in tqdm(range(0, train_images.shape[0])):
    aug_img, aug_point = augment_data(train_images[i], train_labels[i])
    # original data
    train_images_aug.append(train_images[i])
    train_labels_aug.append(train_labels[i])

    # augmented data
    train_images_aug.append(aug_img)
    train_labels_aug.append(aug_point)

# convert to numpy
train_images_aug = np.array(train_images_aug)
train_labels_aug = np.copy(train_labels_aug)

print(train_images_aug.shape)
print(train_labels_aug.shape)

a=1
b=1