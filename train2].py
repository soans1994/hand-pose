"""
Process CMU Hand dataset to get cropped hand datasets.
"""
import os
import numpy as np
import json
from tqdm import tqdm
import glob
import cv2
import numpy as np
#from simple_model import model
from iunet import model
#from iunet2 import model
#from vgg16 import model
#from mobilenetv1 import model
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from PIL import Image

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


id = 150
plot_keypoints(train_images[id], train_labels[id])
"""
fig = plt.figure(figsize=(20,20))
for i in range(16):
    fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_keypoints(train_images[i], train_labels[i])
plt.show()
"""
# train_images = train_images.reshape(train_images.shape[0], 256, 256, 1)
train_images2 = train_images.reshape(train_images.shape[0], 256, 256, 3)
img_height = train_images2.shape[1]
img_width = train_images2.shape[2]
img_channels = train_images2.shape[3]
input_shape = (img_height, img_width, img_channels)
print(input_shape)


def get_model():
    return model(input=input_shape)
    #return model(input_shape=input_shape, num_classes=num_classes)


model = get_model()
# optimizer = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])  # default lr 0.001,1e-3

lr = 1e-3
callbacks = [ModelCheckpoint("test.hdf5", verbose=1, save_best_only=True),
             ReduceLROnPlateau(monitor="val_loss", patience=20, factor=0.1, verbose=1, min_lr=1e-6, ),
             # sdnt go below min_lr
             EarlyStopping(monitor="val_loss", patience=20, verbose=1)]
# history = model.fit(x_train, y_train, batch_size=32, verbose=1, epochs= 500, validation_data=(x_test, y_test), shuffle=False) #callbacks=callbacks)#, class_weight=class_weights )
history = model.fit(train_images, train_labels, batch_size=32, verbose=1, epochs=300, validation_split=0.3,shuffle=False, callbacks=callbacks)
# history = model.fit(x_train, y_train_cat, batch_size=2, verbose=1, epochs= 10, validation_data=(x_test, y_test_cat), shuffle=False)#, class_weight=class_weights )
# shuffle true sshuffles only the training data for every epoch. but may be we need same for checking imporved models.
#model.save("test.hdf5")

# test_images = test_images.reshape(test_images.shape[0], 256, 256, 1)
test_images = test_images.reshape(test_images.shape[0], 256, 256, 3)
_, acc = model.evaluate(test_images, test_labels)
print("Accuracy of test set:", (acc * 100.0), "%")
_, acc = model.evaluate(train_images, train_labels)
print("Accuracy of train set:", (acc * 100.0), "%")

# plot train val acc loss

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "y", label="Training loss")
plt.plot(epochs, val_loss, "r", label="Validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss1.png")
plt.show()

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, "y", label="Training loss")
# plt.plot(epochs, val_loss, "r", label="Validation loss")
plt.plot(epochs, loss, color="#1f77b4", label="Training loss")
plt.plot(epochs, val_loss, color="#ff7f0e", label="Validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")
plt.show()
plt.close()

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
# plt.plot(epochs, acc, "y", label="Training Accuracy")
# plt.plot(epochs, val_acc, "r", label="Validation Accuracy")
plt.plot(epochs, acc, color="#1f77b4", label="Training Accuracy")
plt.plot(epochs, val_acc, color="#ff7f0e", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.show()
plt.close()

fig = plt.figure(figsize=(15, 15))
# make test images keypoints prediction
points_test = model.predict(test_images)
points_train = model.predict(train_images)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_keypoints(test_images[i], np.squeeze(points_test[i]))
    #plot_keypoints(test_images[i], points_test[i])
    plt.show()
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_keypoints(train_images[i], np.squeeze(points_train[i]))
    #plot_keypoints(train_images[i], points_train[i])
    plt.show()

a = 1
