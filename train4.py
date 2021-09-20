"""
Process CMU Hand dataset to get cropped hand datasets.
"""
import os
import pickle
import tensorflow as tf
import keras
import numpy as np
from simple_model import model
#from iunet import model
#from iunet2 import model
#from vgg16 import model
from vgg19 import model
#from vgg162 import model
#from mobilenetv2 import model
#from mobilenetv2 import model
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

#train_images = np.load('xtrain.npy', mmap_mode='r')
#train_labels = np.load('ytrain.npy', mmap_mode='r')
#test_images = np.load('xtest.npy', mmap_mode='r')
#test_labels = np.load('ytest.npy', mmap_mode='r')

train_images = np.load('xtrain.npy')
train_labels = np.load('ytrain.npy')
test_images = np.load('xtest.npy')
test_labels = np.load('ytest.npy')
#train_images = train_images.astype(np.float64)
#test_images = test_images.astype(np.float64)

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

# train_images = train_images.reshape(train_images.shape[0], 256, 256, 1)
train_images2 = train_images.reshape(train_images.shape[0], 256, 256, 3)
img_height = train_images2.shape[1]
img_width = train_images2.shape[2]
img_channels = train_images2.shape[3]
input_shape = (img_height, img_width, img_channels)
num_classes = 42
print(input_shape)

##############adddddddddddddddddd################
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
train_generator = DataGenerator(train_images, train_labels, batch_size = 128)
validation_generator = DataGenerator(test_images, test_labels, batch_size = 128)

def get_model():
    #return model(input=input_shape)
    return model(input_shape=input_shape, num_classes=num_classes)
    #return model(input=input_shape, num_classes=num_classes)


model = get_model()
#optimizer = tf.keras.optimizers.Adam(0.1)
#model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["accuracy"])  # default lr 0.001,1e-3
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])  # default lr 0.001,1e-3

lr = 1e-3
callbacks = [ModelCheckpoint("test.hdf5", verbose=1, save_best_only=True),
             ReduceLROnPlateau(monitor="val_loss", patience=20, factor=0.1, verbose=1, min_lr=1e-6, ),
             # sdnt go below min_lr
             EarlyStopping(monitor="val_loss", patience=20, verbose=1)]
# history = model.fit(x_train, y_train, batch_size=32, verbose=1, epochs= 500, validation_data=(x_test, y_test), shuffle=False) #callbacks=callbacks)#, class_weight=class_weights )
#history = model.fit(train_images, train_labels, batch_size=128, verbose=1, epochs=300, validation_split=0.3,shuffle=False, callbacks=callbacks)#
history = model.fit(train_generator, verbose=1, epochs=300, validation_data=validation_generator,shuffle=False, callbacks=callbacks)#
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
plt.plot(epochs, loss, "y", label="loss")
plt.plot(epochs, val_loss, "r", label="val loss")
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.legend()
plt.savefig("loss1.png")
plt.show()

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, "y", label="Training loss")
# plt.plot(epochs, val_loss, "r", label="Validation loss")
plt.plot(epochs, loss, color="#1f77b4", label="loss")
plt.plot(epochs, val_loss, color="#ff7f0e", label="val loss")
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.legend()
plt.savefig("loss.png")
plt.show()
plt.close()

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
# plt.plot(epochs, acc, "y", label="Training Accuracy")
# plt.plot(epochs, val_acc, "r", label="Validation Accuracy")
plt.plot(epochs, acc, color="#1f77b4", label="acc")
plt.plot(epochs, val_acc, color="#ff7f0e", label="val acc")
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.grid()
plt.legend()
plt.savefig("accuracy.png")
plt.show()
plt.close()

"""
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
"""
a = 1
