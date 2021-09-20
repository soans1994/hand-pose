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
from iunet import model
#from iunet2 import model
#from vgg16 import model
#from mobilenetv2 import model
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from PIL import Image
label=[]
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
    width = 2.2 * B     # This is based on the paper

    # the center of hand box can be
    center = dat["hand_box_center"]
    hand_box = [[center[0] - width / 2., center[1] - width / 2.],
                [center[0] + width / 2., center[1] + width / 2.]]
    hand_box = np.array(hand_box)

    lbl = pts[:, :2] - hand_box[0, :]
    lbl = lbl * 256 / width
    label.append(lbl)


    #lbl = lbl.tolist()
label = np.array(label)
label2 = label.reshape(1912,-1)


def drawKeyPts(im,keyp,keyp2,col,th):
    #for i in keyp:
        #for j in keyp2:
    for i, j in zip(keyp, keyp2):
        x = int(i)
        y = int(j)
        size = 3
        cv2.circle(im,(x,y),size, col,thickness=th)#, lineType=-1)#, shift=0)
    plt.imshow(im)
    return im


size_x = 256
size_y = 256
num_classes = 42

#training list

train_images = []
for img_path in tqdm(sorted(glob.glob("hand_labels/train/crop/*.jpg")),total=1912):
    img = cv2.imread(img_path)
    #img = cv2.resize(img, (size_x, size_y))
    train_images.append(img)

"convert list to np array for ml processing"
train_images = np.array(train_images)#dtype:uint8



#visualize_keypoints(val_images[10], val_target4[10])
x1=label2[400][::2]
y1=label2[400][1::2]
#x2=(512/1920)*x1
#y2=(512/1080)*y1
result = drawKeyPts(train_images[400].copy(),x1,y1,(0,255,0),-1)
#result = drawKeyPts(test[0].copy(),x1,y1,(0,0,255),-1)

from keras.utils import normalize

#val_images = normalize(val_images, axis=1)#uint8 to float64
#label2 = normalize(label2, axis=1)#uint8 to float64
#val_images2 = val_images/255#uint8 to float64
print(train_images.shape)#1912x128x128x3

#added new
train_images = train_images.astype("float32")
label2 = label2.astype("float32")
train_images = train_images/255
label2 = label2/255

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_images, label2, test_size=0.3, random_state=42)


img_height = x_train.shape[1]
img_width = x_train.shape[2]
img_channels = x_train.shape[3]
input_shape = (img_height,img_width,img_channels)
print(input_shape)
#img_height = 256
#img_width = 256
#img_channels = 3
#num_classes = 38
a=1
def get_model():
    return model(input=input_shape)
    #return model(input_shape=input_shape, num_classes=num_classes)

model = get_model()
#optimizer = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])#default lr 0.001,1e-3
#model.compile(optimizer="adam", loss="mean_absolute_error", metrics=["accuracy"])#default lr 0.001,1e-3
#model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])#default lr 0.001,1e-3
#model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
#model.summary()

lr = 1e-3
callbacks = [ModelCheckpoint("test.hdf5", verbose=1, save_best_only=True),
                 ReduceLROnPlateau(monitor="val_loss", patience=20, factor=0.1, verbose=1, min_lr=1e-6,),#sdnt go below min_lr
                EarlyStopping(monitor="val_loss", patience=20, verbose=1)]
history = model.fit(x_train, y_train, batch_size=32, verbose=1, epochs= 300, validation_data=(x_test, y_test), shuffle=True, callbacks=callbacks)#, class_weight=class_weights )
#history = model.fit(x_train, y_train_cat, batch_size=2, verbose=1, epochs= 10, validation_data=(x_test, y_test_cat), shuffle=False)#, class_weight=class_weights )
#shuffle true sshuffles only the training data for every epoch. but may be we need same for checking imporved models.
#model.save("mobilenetv2false300.hdf5")
_, acc = model.evaluate(x_test, y_test)
print("Accuracy:", (acc*100.0),"%")

#plot train val acc loss

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


acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
#plt.plot(epochs, acc, "y", label="Training Accuracy")
#plt.plot(epochs, val_acc, "r", label="Validation Accuracy")
plt.plot(epochs, acc, color="#1f77b4", label="Training Accuracy")
plt.plot(epochs, val_acc, color="#ff7f0e", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.show()
plt.close()

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
#plt.plot(epochs, loss, "y", label="Training loss")
#plt.plot(epochs, val_loss, "r", label="Validation loss")
plt.plot(epochs, loss, color="#1f77b4", label="Training loss")
plt.plot(epochs, val_loss, color="#ff7f0e", label="Validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")
plt.show()
plt.close()
