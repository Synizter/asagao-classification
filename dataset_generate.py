from statistics import median
from unittest.mock import _patch_dict
from weakref import ref
import cv2
from cv2 import CC_STAT_AREA
import numpy as np
from glob import glob
import math
import matplotlib.pyplot as plt
from tiled_detiled import detiled_image, tiled_image

t = []
def cal_crop_size(w,h, target)->tuple:
    coeff_h = math.ceil(h / target)
    coeff_w = math.ceil(w / target)
    return (target * coeff_w, target * coeff_h)

nn = 1
def div_img_and_save(crop, mask, patch_size = 200):
    global nn
    for i in range(0,crop.shape[0],patch_size):
        for j in range(0,crop.shape[1],patch_size):
            cc_data = crop[i:i + patch_size, j:j + patch_size, :] # all ch
            cc_mask = mask[i:i + patch_size, j:j + patch_size] # only bw
            if(cc_data.shape[0] != cc_data.shape[1] or cc_data.shape[0] < patch_size or cc_data.shape[1] < patch_size): 
                continue
            
            if(np.sum(cc_mask) > 10027008.0):
                cv2.imwrite("dataset/validation/asagao/asagao_mask_%s.png" % str(nn).zfill(4), cc_mask)
                cv2.imwrite("dataset/train/asagao/asagao_%s.png" % str(nn).zfill(4), cc_data)
                print("Asagao tile no. %d saved" % nn)

                nn += 1

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



#ASAGAO
def asagao_save():
    # for mask,img in zip(mask_path, image_path):
    mask_path = ['DJI_20211208095102_0006_mask.jpg',
    'DJI_20211208095108_0010_mask.jpg',
    'DJI_20211208095119_0019_mask.jpg',
    'DJI_20211208102326_0003_mask.JPG',
    'DJI_20211208102338_0009_mask.JPG']

    image_path = ['dataset/raw/not_labeled/30m-1/DJI_20211208095102_0006.JPG',
    'dataset/raw/not_labeled/30m-1/DJI_20211208095108_0010.JPG',
    'dataset/raw/not_labeled/30m-1/DJI_20211208095119_0019.JPG',
    'dataset/raw/not_labeled/50m/DJI_20211208102326_0003.JPG',
    'dataset/raw/not_labeled/50m/DJI_20211208102338_0009.JPG']

    for fmsk, fimg in zip(mask_path, image_path):
        ref_mask = cv2.imread(fmsk)
        image = cv2.imread(fimg)

        # find countour location on the reference mask
        contours = cv2.findContours(ref_mask[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # #check the result
        for c in contours:
            # draw a rectangle
            rect = cv2.boundingRect(c)
            if rect[2] < 30 or rect[3] < 30: continue
            x,y,w,h = rect
            # cv2.rectangle(ref_mask, (x,y), (x+w,y+h), (255, 0, 0), 10) # draw a rectangle center from contour
            

            new_w, new_h = cal_crop_size(w, h, 256) #the orignal contour size is varied, calculate new crop size to get all data
                
            cc_img = image[y:y + new_h, x: x + new_w,:] #cropped image from orignal
            cc_msk = ref_mask[y:y + new_h, x: x + new_w] # crop image from mask
            
            # plt.subplot(211),plt.imshow(cc_img)
            # plt.subplot(212),plt.imshow(cc_msk)
            # plt.show()

            # print("Image width | heigth (px) : {}|{}".format(new_w, new_h))
            # print("number of expected tiles : {}".format((new_w * new_h) / (256*256)))
            div_img_and_save(cc_img, cc_msk, 256)
import time
# None Asagao
def non_asagao_save():
    mask_path = ['DJI_20211208095102_0006_mask.jpg',
    'DJI_20211208095108_0010_mask.jpg',
    'DJI_20211208095119_0019_mask.jpg',
    'DJI_20211208102326_0003_mask.JPG',
    'DJI_20211208102338_0009_mask.JPG']

    image_path = ['dataset/raw/not_labeled/30m-1/DJI_20211208095102_0006.JPG',
    'dataset/raw/not_labeled/30m-1/DJI_20211208095108_0010.JPG',
    'dataset/raw/not_labeled/30m-1/DJI_20211208095119_0019.JPG',
    'dataset/raw/not_labeled/50m/DJI_20211208102326_0003.JPG',
    'dataset/raw/not_labeled/50m/DJI_20211208102338_0009.JPG']

    nbr = 1
    for fmsk, fimg in zip(mask_path, image_path):
        # read the mask
        ref_mask = cv2.imread(fmsk)
        image = cv2.imread(fimg)

        patch_size = 256
        tic = time.time()
        print("Writing non-asagao pixel in %s" % fimg)
        #tile all image
        for i in range(0,ref_mask.shape[0],patch_size):
            for j in range(0,ref_mask.shape[1],patch_size):
                # crop (pad if needed) the image
                cc_data = ref_mask[i:i + patch_size, j:j + patch_size, :] # all ch
                if cc_data.shape[0] < patch_size: #patch heigth
                    padding = patch_size - cc_data.shape[0]
                    patch = np.zeros((padding ,cc_data.shape[1],3))
                    # print("Heigth Stack",cc_data.shape, patch.shape, nn)    
                    cc_data = np.concatenate((cc_data, patch), axis=0)
                if cc_data.shape[1] < patch_size: #patch width
                    padding = patch_size - cc_data.shape[1]
                    patch = np.zeros((cc_data.shape[0] ,padding, 3))
                    # print("Width Stack",cc_data.shape, patch.shape, nn)  
                    cc_data = np.concatenate((cc_data, patch), axis=1)

                if(np.sum(cc_data) > 0):
                    continue

                else:
                    cv2.imwrite("dataset/validation/non_asagao/non_asagao_mask_%s.png" % str(nbr).zfill(4), cc_data)
                
                # crop (pad if needed) the mask
                cc_data = image[i:i + patch_size, j:j + patch_size, :] # all ch
                if cc_data.shape[0] < patch_size: #patch heigth
                    padding = patch_size - cc_data.shape[0]
                    patch = np.zeros((padding ,cc_data.shape[1],3))
                    # print("Heigth Stack",cc_data.shape, patch.shape, nn)    
                    cc_data = np.concatenate((cc_data, patch), axis=0)
                if cc_data.shape[1] < patch_size: #patch width
                    padding = patch_size - cc_data.shape[1]
                    patch = np.zeros((cc_data.shape[0] ,padding, 3))
                    # print("Width Stack",cc_data.shape, patch.shape, nn)  
                    cc_data = np.concatenate((cc_data, patch), axis=1)

                cv2.imwrite("dataset/train/non_asagao/non_asagao_%s.png" % str(nbr).zfill(4), cc_data)

                nbr+=1
    
        toc = time.time()
        print("Done in %.8f" % (toc - tic))
            


non_asagao_save()
