from tkinter import dialog
import cv2
from cv2 import resize
import numpy as np
from glob import glob


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

def segment(file:str, low:np.array, high:np.array, fname = None):
    image = cv2.imread(file) #replace with file
    if image is None:
        print("No image opened")
        return
    mask = cv2.inRange(image, low, high)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    dilated = cv2.dilate(mask, kernel)

    # # find  closed contour and fill
    cnt = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0] if len(cnt) == 2 else cnt[1]

    nbr_of_cnt = 0
    for c in cnt:
        cv2.drawContours(dilated, [c], 0, (255, 255, 255), -1) # fill close area of contour
        rect = cv2.boundingRect(c)
        if rect[2] > 30 or rect[3] > 30: nbr_of_cnt += 1

    # #convert ot 0/1
    non_white = np.where(dilated != 255)
    dilated[non_white] = 0
    print("find %d contour" % nbr_of_cnt)

    if(fname is not None):
        cv2.imwrite(fname, dilated)
    else:
        cv2.imshow("Image", resizeWithAspectRatio(image, 800))
        cv2.imshow("Mask", resizeWithAspectRatio(dilated, 800))
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    lower = np.array([0, 0, 105], dtype="uint8")
    upper = np.array([60, 90, 255], dtype="uint8")
    segment('dataset/raw/labeled/10m/DJI_0020100.tif', lower ,upper, "DJI_0020100_mask.jpg")
