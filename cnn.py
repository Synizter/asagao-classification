import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.utils import normalize

# dataset = []

# img = cv2.imread('Capture.PNG')
# # cv2.imshow("TEST", img)

# image = Image.fromarray(img)
# image = np.array(image)


# print(image.shape)
# print("data type :", image.dtype)
# print("Max/Max value: {}/{}".format(image.max(), image.min()))
# #convert to float32
# image = image / 255.0

# print(image[:,:,2])

from glob import glob
f_img = sorted(glob('dataset/train/*'))

image = cv2.imread(f_img[0])
print(type(image), image.shape, image.dtype, image.max(), image.min())
image = image.astype('float32') / 255
print(type(image), image.shape, image.dtype, image.max(), image.min())

# cv2.imshow("test", np.array(image))
# cv2.waitKey(0)
# cv2.destroyAllWindows()