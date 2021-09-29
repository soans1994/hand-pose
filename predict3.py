
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
#from vgg16 import model
#from mobilenetv2 import model
#from simple_model2 import model
from matplotlib import pyplot as plt
from load_numpy_data2 import generator
from load_numpy_dataforold import generator


#################ADD###############
#samples = sorted(glob.glob("hand_labels_synth/synth3/*.jpg"))
#samples2 = sorted(glob.glob("hand_labels_synth/synth3val/*.jpg"))
samples = sorted(glob.glob("hand_labels/train/crop/*.jpg"))
samples2 = sorted(glob.glob("hand_labels/test/crop/*.jpg"))
#train_images, train_labels = generator(samples, batch_size=32, aug=None)#, aug=train_aug)
train_generator = generator(samples, batch_size=64, aug=None)#, aug=train_aug)
validation_generator = generator(samples2, batch_size=64, aug=None)#, aug=train_aug)
#input_shape = (368, 368, 3)
input_shape = (256, 256, 3)
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

for x_train,y_train in validation_generator:
    print(x_train.shape, y_train.shape)
    print(x_train[0].shape, y_train[0].shape)
    break
for x_test,y_test in validation_generator:
    print(x_test.shape, y_test.shape)
    print(x_test[0].shape, y_test[0].shape)
    break
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
print(y_test_pred.shape, y_train_pred.shape)
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
    plot_keypoints(x_test[i], np.squeeze(y_test_pred[i]))
    #plot_keypoints(test_images[i], points_test[i])
    plt.show()
#for i in range(4):
    #ax = fig.add_subplot(2, 2, i + 1, xticks=[], yticks=[])
    #plot_keypoints(x_test[i], np.squeeze(y_test[i]))
    #plot_keypoints(test_images[i], points_test[i])
    #plt.show()

for i in range(4):
    ax = fig.add_subplot(2, 2, i + 1, xticks=[], yticks=[])
    plot_keypoints(x_train[i], np.squeeze(y_train_pred[i]))
    #plot_keypoints(train_images[i], points_train[i])
    plt.show()
#for i in range(4):
    #ax = fig.add_subplot(2, 2, i + 1, xticks=[], yticks=[])
    #plot_keypoints(x_train[i], np.squeeze(y_train[i]))
    #plot_keypoints(train_images[i], points_train[i])
    #plt.show()

a=1
