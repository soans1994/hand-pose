
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

#resize images if needed

size_x = 256
size_y = 256
#num_classes = 14
num_classes = 42#38

#training list

train_images = []
for img_path in tqdm(sorted(glob.glob("hand_labels/train/crop/*.jpg")),total=1912):
    img = cv2.imread(img_path)
    #img = cv2.resize(img, (size_x, size_y))
    train_images.append(img)

"convert list to np array for ml processing"
train_images = np.array(train_images)#dtype:uint8
#val_images2 = val_images/255#uint8 to float64
#from keras.utils import normalize
#train_images = np.expand_dims(train_images, axis=3)#701x256x256 to 701x256x256x3
######train_images = train_images.astype(np.float32)#fks up output
#val_images = normalize(val_images, axis=1)#uint8 to float64



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_images, label2, test_size=0.3, random_state=42)
#x_train 490,256,256,3 x_test 211,256,256,3 y_train 490,256,256 y_test 211,256,256


img_height = x_train.shape[1]
img_width = x_train.shape[2]
img_channels = x_train.shape[3]
input_shape = (img_height,img_width,img_channels)
print(input_shape)

def get_model():
    return model(input=input_shape)
    #return model(input_shape=input_shape, num_classes=num_classes)

model = get_model()
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
model.summary()
model.load_weights("test.hdf5") #Accuracy: 91.47552847862244 % Mean IOU : 0.561591
#model.load_weights("unet2mse.h5") #Accuracy: 86.97680234909058 % Mean IOU : 0.4135831

_, acc = model.evaluate(x_test, y_test)
print("Accuracy:", (acc*100.0),"%")

y_pred = model.predict(x_test)
print(y_pred.shape)#(211, 128, 128, 38) prob 0 to 1
#y_pred_argmax = np.argmax(y_pred, axis=2)
#print(y_pred_argmax.shape)#(211, 128, 128)# argmax returns classs of max prob


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

#######read single image for test resize##########
img = cv2.imread("hand_labels/train/data/012637014_01_l.jpg")
#img = cv2.resize(img, (size_x, size_y))
img = np.array(img)#dtype:uint8

#visualize_keypoints(val_images[10], val_target4[10])
x1=y_test[10][::2]
y1=y_test[10][1::2]
x2=y_pred[10][::2]
y2=y_pred[10][1::2]
x_ratio = size_x / img.shape[1]
y_ratio = size_y / img.shape[0]
x3 = y_pred[10][::2]* x_ratio
y3 = y_pred[10][1::2] * y_ratio
#x2=(512/1920)*x1
#y2=(512/1080)*y1
#x_test = x_test*255
original = drawKeyPts(x_test[10].copy(),x1,y1,(0,255,0),-1)
result = drawKeyPts(x_test[10].copy(),x2,y2,(0,255,0),-1)
resize = drawKeyPts(img.copy(),x3,y3,(0,255,0),-1)
a=1
