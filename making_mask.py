from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
mask =  cv2.imread('./mask.png',cv2.IMREAD_GRAYSCALE)
image = cv2.imread('./train_source_image/TRAIN_SOURCE_0000.png',cv2.IMREAD_COLOR)
image = cv2.resize(image,dsize=(960,540),interpolation=cv2.INTER_LINEAR)
dst = cv2.imread('./mask.png',cv2.IMREAD_COLOR)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

cv2.copyTo(image,mask,dst)
plt.imshow(dst)
plt.show()

