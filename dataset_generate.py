import cv2 as cv
from cv2 import CC_STAT_AREA
import numpy as np
from glob import glob
import math
import matplotlib.pyplot as plt

def resizeWithAspectRatio(im, w=None, h=None, inter=cv.INTER_AREA):
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

    return cv.resize(im, dim, interpolation=inter)

def generate_mask(f_img_list, f_lab_list, target_size):
  #delete Yellow line in image
    lower = np.array([0, 180, 250], dtype="uint8")
    upper = np.array([100, 255, 255], dtype="uint8")
    n = 1

    for im, lab in zip(f_img_list, f_lab_list):
        print("Start processing Image {} (label file {})".format(im, lab))
        img_lab = cv.imread(im)
        img_ori = cv.imread(lab)
        
        mask = cv.inRange(img_lab, lower, upper)

        # # find  closed contour and fill
        cnt = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnt = cnt[0] if len(cnt) == 2 else cnt[1]

        nbr_of_cnt = 0
        for c in cnt:
            cv.drawContours(mask, [c], 0, (255, 255, 255), -1) # fill close area of contour
            rect = cv.boundingRect(c)
            if rect[2] > 100 or rect[3] > 100: nbr_of_cnt += 1
        crop = cv.bitwise_and(img_ori, img_ori, mask=mask) #create mask area of Asago
        cv.imwrite("actual_location_" + im[-35:], mask)
        n+= 1
        # cv.waitKey(0)
        # cv.destroyAllWindows()
def generate_dataset(f_img_list, f_lab_list, target_size):
    #delete Yellow line in image
    lower = np.array([0, 180, 250], dtype="uint8")
    upper = np.array([100, 255, 255], dtype="uint8")    
    

    for im, lab in zip(f_img_list, f_lab_list):
        print("Start processing Image {} (label file {})".format(im, lab))
        img_lab = cv.imread(im)
        img_ori = cv.imread(lab)
        
        mask = cv.inRange(img_lab, lower, upper)

        # cv.imshow("TEST",resizeWithAspectRatio( img_lab, 800))
        # cv.imshow("TSET", resizeWithAspectRatio(mask, 800))
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # # find  closed contour and fill
        cnt = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnt = cnt[0] if len(cnt) == 2 else cnt[1]

        nbr_of_cnt = 0
        for c in cnt:
            cv.drawContours(mask, [c], 0, (255, 255, 255), -1) # fill close area of contour
            rect = cv.boundingRect(c)
            if rect[2] > 100 or rect[3] > 100: nbr_of_cnt += 1
        # crop = cv.bitwise_and(img_ori, img_ori, mask=mask) #create mask area of Asago

        cv.imshow("mask",resizeWithAspectRatio(mask, 800))
        cv.imshow("Label",resizeWithAspectRatio(img_lab, 800))

        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # print("------ Found contour {} (offset = 100px^2)".format(nbr_of_cnt))
        # print("------ start generating dataset from contour")
        # i = 1


        # # cv.imshow("Labeled Image", resizeWithAspectRatio(img_lab, 600))
        # # cv.imshow("Mask", resizeWithAspectRatio(mask, 600))
        # for c in cnt:
        #     # draw a rectangle
        #     rect = cv.boundingRect(c)
        #     if rect[2] < 100 or rect[3] < 100: continue
        #     x,y,w,h = rect

        #     new_w, new_h = cal_crop_size(w,h,target_size) #the orignal contour size is varied, calculate new crop size to get all data
        #     cc_img = img_ori[y:y + new_h, x: x + new_w,:] #cropped image from orignal
        #     cc_msk = mask[y:y + new_h, x: x + new_w] # crop image from mask
            
        #     # plt.subplot(211),plt.imshow(cc_img)
        #     # plt.subplot(212),plt.imshow(cc_msk)
        #     # plt.show()

        #     # cv.imshow("crop", resizeWithAspectRatio(cc_img, 600))
        #     # cv.imshow("crop_mask", resizeWithAspectRatio(cc_msk, 600))
        #     # cv.waitKey(0)
            
        #     div_img_and_save(cc_img, cc_msk, target_size) #Divide image accordingly to sqaure patch size *default 200 (200x200px)
            
        #     i+=1
        # # cv.destroyAllWindows()

def cal_crop_size(w,h, target)->tuple:
    coeff_h = math.ceil(h / target)
    coeff_w = math.ceil(w / target)
    return (target * coeff_w, target * coeff_h)
    

nn = 1
# #input, cropped image
def div_img_and_save(crop, mask, patch_size = 200):
    global nn
    for i in range(0,crop.shape[0],patch_size):
        
        for j in range(0,crop.shape[1],patch_size):
            cc_data = crop[i:i + patch_size, j:j + patch_size, :] # all ch
            cc_mask = mask[i:i + patch_size, j:j + patch_size] # only bw
            if(cc_data.shape[0] != cc_data.shape[1] or cc_data.shape[0] < patch_size or cc_data.shape[1] < patch_size): continue
            cv.imwrite("dataset/validation/asagao_mask_%s.png" % str(nn).zfill(4), cc_mask)
            cv.imwrite("dataset/train/asagao_%s.png" % str(nn).zfill(4), cc_data)
            print("Dataset patches {} of size {} px saved".format(nn, patch_size))
            nn += 1

if __name__ == "__main__":
    import os
    # remove previos save dataset

    FILES_LABELS_IMG = sorted(glob('dataset/raw/labeled/30m-1/*')) #get image with label mask
    FILES_ORIG_IMG = sorted(glob('dataset/raw/not_labeled/30m-1/*'))

    # uncomment to include 50m data
    FILES_LABELS_IMG += sorted(glob('dataset/raw/labeled/50m/*')) #get image with label mask
    FILES_ORIG_IMG += sorted(glob('dataset/raw/not_labeled/50m/*'))

    if len(FILES_LABELS_IMG) > 0 and len(FILES_ORIG_IMG) > 0 and len(FILES_ORIG_IMG) == len(FILES_LABELS_IMG):
        print("Program start!")
        # if len(os.listdir('dataset/train')) != 0:
        #     print("Found previous dataset, attemp delete")
        #     file_to_delete = os.listdir('dataset/train')
        #     for f in file_to_delete:
        #         os.remove('dataset/train/' + f)
        
        # if len(os.listdir('dataset/validation')) != 0:
        #     print("Found previous dataset, attemp delete")
        #     file_to_delete = os.listdir('dataset/validation')
        #     for f in file_to_delete:
        #         os.remove('dataset/validation/' + f)
        generate_dataset(FILES_LABELS_IMG, FILES_ORIG_IMG, 256)

        # generate_mask(FILES_LABELS_IMG, FILES_ORIG_IMG, 256)
    else:
        print("cannot find target file")
        print(FILES_LABELS_IMG, len(FILES_LABELS_IMG))
        print(FILES_ORIG_IMG, len(FILES_ORIG_IMG))




    



# #\\wsl$\Ubuntu-18.04\home\capilab\proj\opencv_asagau