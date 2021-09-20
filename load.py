import numpy
from pycocotools.coco import COCO
import pandas as pd
from tqdm import tqdm
import glob
import cv2
import numpy as np
from iunet import model
from matplotlib import pyplot as plt


size_x = 512
size_y = 512
num_classes = 42
...

train_annot_path = 'hand_labels/annotations/panoptic_train.json'
val_annot_path = 'hand_labels/annotations/panoptic_test.json'
train_coco = COCO(train_annot_path) # load annotations for training set
val_coco = COCO(val_annot_path) # load annotations for validation set
...
# function iterates ofver all ocurrences of a  person and returns relevant data row by row
def get_meta(coco):
    ids = list(coco.imgs.keys())
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # basic parameters of an image
        img_file_name = img_meta['file_name']
        w = img_meta['width']
        h = img_meta['height']
        # retrieve metadata for all persons in the current image
        anns = coco.loadAnns(ann_ids)

        yield [img_id, img_file_name, w, h, anns]

...

# iterate over images
for img_id, img_fname, w, h, meta in get_meta(train_coco):
    ...
    # iterate over all annotations of an image
    for m in meta:
        # m is a dictionary
        keypoints = m['keypoints']

######################################
def convert_to_df(coco):
    images_data = []
    persons_data = []
    # iterate over all images
    for img_id, img_fname, w, h, meta in get_meta(coco):
        images_data.append({
            'image_id': int(img_id),
            'path': img_fname,
            'width': int(w),
            'height': int(h)
        })
        # iterate over all metadata
        for m in meta:
            persons_data.append({
                'image_id': m['image_id'],
                'is_crowd': m['iscrowd'],
                'bbox': m['bbox'],
                'area': m['area'],
                'keypoints': m['keypoints'],
            })
    # create dataframe with image paths
    images_df = pd.DataFrame(images_data)
    images_df.set_index('image_id', inplace=True)
    # create dataframe with persons
    persons_df = pd.DataFrame(persons_data)
    persons_df.set_index('image_id', inplace=True)
    return images_df, persons_df
##################################3
"""
images_df, persons_df = convert_to_df(train_coco)
train_coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
train_coco_df['source'] = 0
print(train_coco_df.head())
print(len(images_df), len(persons_df))
print(images_df.shape, persons_df.shape)
print(len(train_coco_df))
print(train_coco_df.shape)
train_target = train_coco_df.pop('keypoints')
print(len(train_target), train_target)
"""

images_df, persons_df = convert_to_df(val_coco)
val_coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
val_coco_df['source'] = 0
print(val_coco_df.head())
print(len(images_df), len(persons_df))
print(images_df.shape, persons_df.shape)
print(len(val_coco_df))
print(val_coco_df.shape)
#val_target = val_coco_df.pop('keypoints')
numeric_feature_names = ['keypoints']
val_target = val_coco_df[numeric_feature_names]
val_target.head()
print(len(val_target), val_target)
val_target2=val_target.to_numpy()
#val_target3=numpy.array(val_target2)
val_target3 = np.array(val_coco_df['keypoints'].tolist())
val_target3 = numpy.delete(val_target3, numpy.s_[2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,59,62], axis=1)
val_target4 = val_target3 / size_x

#val_target3 = np.reshape(val_target2, (-1, 2))
#val_target4=np.array(val_target.values.tolist())#846x63

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



size_x = 512
size_y = 512
num_classes = 42

#training list

val_images = []
for img_path in tqdm(sorted(glob.glob("hand_labels/manual_test/*.jpg")),total=846):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (size_x, size_y))
    val_images.append(img)

"convert list to np array for ml processing"
val_images = np.array(val_images)#dtype:uint8

test = []
img_path = "hand_labels/manual_test/000835470_01_r.jpg"
img = cv2.imread(img_path)
#img = cv2.resize(img, (size_x, size_y))
test.append(img)
test=np.array(test)

#visualize_keypoints(val_images[10], val_target4[10])
x1=val_target3[1][::2]
y1=val_target3[1][1::2]
x2=(512/1920)*x1
y2=(512/1080)*y1
result = drawKeyPts(val_images[1].copy(),x2,y2,(0,255,0),-1)
#result = drawKeyPts(test[0].copy(),x1,y1,(0,0,255),-1)

from keras.utils import normalize
#train_images = np.expand_dims(train_images, axis=3)#701x256x256 to 701x256x256x3
#####train_images = train_images.astype(np.float32)
val_images = normalize(val_images, axis=1)#uint8 to float64
print(val_images.shape)#1912x128x128x3

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(val_images, val_target3, test_size=0.3, random_state=42)


img_height = x_train.shape[1]
img_width = x_train.shape[2]
img_channels = x_train.shape[3]
input_shape = (img_height,img_width,img_channels)
print(input_shape)
#img_height = 256
#img_width = 256
#img_channels = 3
#num_classes = 38

def get_model():
    return model(input=input_shape)

model = get_model()
#optimizer = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])#default lr 0.001
#model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
#model.summary()

history = model.fit(x_train, y_train, batch_size=32, verbose=1, epochs= 500, validation_data=(x_test, y_test), shuffle=False)#, class_weight=class_weights )
#history = model.fit(x_train, y_train_cat, batch_size=2, verbose=1, epochs= 10, validation_data=(x_test, y_test_cat), shuffle=False)#, class_weight=class_weights )
#shuffle true sshuffles only the training data for every epoch. but may be we need same for checking imporved models.
model.save("new.hdf5")
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
plt.savefig("loss.png")
plt.show()


acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "y", label="Training Accuracy")
plt.plot(epochs, val_acc, "r", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.show()


