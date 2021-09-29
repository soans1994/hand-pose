
import json
from tqdm import tqdm
import glob
import cv2
import keras
import numpy as np
#from iunet import model
#from iunet2 import model
#from vgg19 import model
from simplevgg import model
#from mobilenetv2 import model
#from simple_model2 import model
from matplotlib import pyplot as plt
import pickle

"""
pickle_in = open("x_train_aug2.pickle","rb")
train_images = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open("y_train_aug2.pickle","rb")
train_labels = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("x_test.pickle","rb")
test_images = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open("y_test.pickle","rb")
test_labels = pickle.load(pickle_in)
pickle_in.close()

x_train = train_images
y_train = train_labels
x_test = test_images
y_test = test_labels
"""
"""
train_images = np.load('xtrain.npy', mmap_mode='r')
train_labels = np.load('ytrain.npy', mmap_mode='r')
test_images = np.load('xtest.npy', mmap_mode='r')
test_labels = np.load('ytest.npy', mmap_mode='r')
"""
x_train = np.load('x_train.pickle', allow_pickle=True)
y_train = np.load('y_train.pickle', allow_pickle=True)
x_test = np.load('x_test.pickle', allow_pickle=True)
y_test = np.load('y_test.pickle', allow_pickle=True)

#x_train = train_images
#y_train = train_labels
#x_test = test_images
#y_test = test_labels
img_height = x_train.shape[1]
img_width = x_train.shape[2]
img_channels = x_train.shape[3]

input_shape = (img_height,img_width,img_channels)
print(input_shape)
num_classes = 42

class DataGenerator(keras.utils.Sequence):
  def __init__(self, x_data, y_data, batch_size):
    self.x, self.y = x_data, y_data
    self.batch_size = batch_size
    self.num_batches = np.ceil(len(x_data) / batch_size)
    self.batch_idx = np.array_split(range(len(x_data)), self.num_batches)

  def __len__(self):
    return len(self.batch_idx)

  def __getitem__(self, idx):
    batch_x = self.x[self.batch_idx[idx]]
    batch_y = self.y[self.batch_idx[idx]]
    return batch_x, batch_y

#################ADD###############
train_generator = DataGenerator(x_train, y_train, batch_size = 32)#119 batches(3824/32)
validation_generator = DataGenerator(x_test, y_test, batch_size = 32)
def get_model():
    return model(input_shape)
    #return model(input=input_shape)
    #return model(input_shape=input_shape, num_classes=num_classes)

model = get_model()
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
#model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.load_weights("test.hdf5") #Accuracy: 91.47552847862244 % Mean IOU : 0.561591
#model.load_weights("unet2mse.h5") #Accuracy: 86.97680234909058 % Mean IOU : 0.4135831

#_, acc = model.evaluate(x_train, y_train)
_, acc = model.evaluate(train_generator)
print("Train Accuracy evaluate:", (acc*100.0),"%")
#_, acc = model.evaluate(x_test, y_test)
_, acc = model.evaluate(validation_generator)
print("Test Accuracy:", (acc*100.0),"%")

#_, acc = model.evaluate(x_train.reshape(x_train.shape[0], 256, 256, 3), y_train)
#print("Train Accuracy evaluate:", (acc*100.0),"%")
#_, acc = model.evaluate(x_test.reshape(x_test.shape[0], 256, 256, 3), y_test)
#print("Test Accuracy:", (acc*100.0),"%")

#y_test_pred = model.predict(x_test)
y_test_pred = model.predict(validation_generator[0][0])#[0][0]is data and [1][0] is label
print(y_test_pred.shape)#(211, 128, 128, 38) prob 0 to 1
#y_train_pred = model.predict(x_train)
y_train_pred = model.predict(train_generator[0][0])
print(y_train_pred.shape)#(211, 128, 128, 38) prob 0 to 1


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

fig = plt.figure(figsize=(15, 15))
for i in range(4):
    ax = fig.add_subplot(2, 2, i + 1, xticks=[], yticks=[])
    plot_keypoints(x_test[i], np.squeeze(y_train_pred[i]))
    #plot_keypoints(test_images[i], points_test[i])
    plt.show()
for i in range(4):
    ax = fig.add_subplot(2, 2, i + 1, xticks=[], yticks=[])
    plot_keypoints(x_test[i], np.squeeze(y_test[i]))
    #plot_keypoints(test_images[i], points_test[i])
    plt.show()

for i in range(4):
    ax = fig.add_subplot(2, 2, i + 1, xticks=[], yticks=[])
    plot_keypoints(x_train[i], np.squeeze(y_test_pred[i]))
    #plot_keypoints(train_images[i], points_train[i])
    plt.show()
for i in range(4):
    ax = fig.add_subplot(2, 2, i + 1, xticks=[], yticks=[])
    plot_keypoints(x_train[i], np.squeeze(y_train[i]))
    #plot_keypoints(train_images[i], points_train[i])
    plt.show()

a=1
