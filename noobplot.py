import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
dat = json.load(open("0001.json"))
pts = np.array(dat['hand_pts'])
img = cv2.imread("0001.jpg")
plt.imshow(img, cmap='gray')
plt.scatter(pts[:,0], pts[:,1])
plt.show()