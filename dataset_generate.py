from weakref import ref
import cv2
from cv2 import CC_STAT_AREA
import numpy as np
from glob import glob
import math
import matplotlib.pyplot as plt

def cal_crop_size(w,h, target)->tuple:
    coeff_h = math.ceil(h / target)
    coeff_w = math.ceil(w / target)
    return (target * coeff_w, target * coeff_h)

def resizeWithAspectRatio(im, w=None, h=None, inter=cv2.INTER_AREA):
    dim = None

    (origh,origw) = im.shape[:2]
    if w is None and h is None:
        raise "Error: please specify either width or heigth"
    elif w is None:
        r = h / float(origh)
        dim = (int(origw * r), h)
    elif h is None:
        r = w / float(origw)
        dim = (w, int(origh * r))

    return cv2.resize(im, dim, interpolation=inter)


ref_mask = cv2.imread('DJI_0012000_mask.tif')
# ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2GRAY)

# image = cv2.imread('dataset/raw/not_labeled/30m-1/DJI_20211208095108_0010.JPG')


# find countour location on the reference mask
contours = cv2.findContours(ref_mask[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# #check the result

for c in contours:
    # draw a rectangle
    rect = cv2.boundingRect(c)
    if rect[2] < 30 or rect[3] < 30: continue
    x,y,w,h = rect

    print("Found!")
    cv2.rectangle(ref_mask, (x,y), (x+w,y+h), (255, 0, 0), 10)
    

cv2.imshow("crop", resizeWithAspectRatio(ref_mask, 600))
cv2.waitKey(0)
cv2.destroyAllWindows()

