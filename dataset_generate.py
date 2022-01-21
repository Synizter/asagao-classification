import cv2 as cv
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

def generate_dataset(f_img_list, f_lab_list, target_size):
    #delete Yellow line in image
    lower = np.array([0, 180, 250], dtype="uint8")
    upper = np.array([100, 255, 255], dtype="uint8")
    

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

        # cv.imshow("mask",resizeWithAspectRatio(crop, 800))
        # cv.imshow("Label",resizeWithAspectRatio(img_lab, 800))

        # cv.waitKey(0)
        # cv.destroyAllWindows()

        print("------ Found contour {} (offset = 100px^2)".format(nbr_of_cnt))
        print("------ start generating dataset from contour")
        i = 1
 
        for c in cnt:
            # draw a rectangle
            rect = cv.boundingRect(c)
            if rect[2] < 100 or rect[3] < 100: continue
            x,y,w,h = rect

            new_w, new_h = cal_crop_size(w,h,target_size) #the orignal contour size is varied, calculate new crop size to get all data
            cc_img = img_ori[y:y + new_h, x: x + new_w,:] #cropped image from orignal
            cc_msk = mask[y:y + new_h, x: x + new_w] # crop image from mask
            
            # plt.subplot(211),plt.imshow(cc_img)
            # plt.subplot(212),plt.imshow(cc_msk)
            # plt.show()
            div_img_and_save(cc_img, cc_msk, target_size) #Divide image accordingly to sqaure patch size *default 200 (200x200px)
            
            i+=1

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
            nn += 1
            cv.imwrite("dataset/validation/asagao_mask_%d.png" % nn, cc_mask)
            cv.imwrite("dataset/train/asagao_%d.png" % nn, cc_data)
            print("Dataset patches {} of size {} px saved".format(nn, patch_size))

if __name__ == "__main__":
    import os
    # remove previos save dataset

    if len(os.listdir('dataset/train')) != 0:
        print("Found previous dataset, attemp delete")
        file_to_delete = os.listdir('dataset/train')
        for f in file_to_delete:
            os.remove('dataset/train/' + f)
    
    if len(os.listdir('dataset/validation')) != 0:
        print("Found previous dataset, attemp delete")
        file_to_delete = os.listdir('dataset/validation')
        for f in file_to_delete:
            os.remove('dataset/validation/' + f)

    FILES_LABELS_IMG = sorted(glob('dataset/raw/labeled/*')) #get image with label mask
    FILES_ORIG_IMG = list(sorted(set(glob('dataset/raw/not_labeled/*')) - set(FILES_LABELS_IMG)))

    if len(FILES_LABELS_IMG) > 0 and len(FILES_ORIG_IMG) > 0 and len(FILES_ORIG_IMG) == len(FILES_LABELS_IMG):
        generate_dataset(FILES_LABELS_IMG, FILES_ORIG_IMG, 256)
    else:
        print("cannot find target file")
        print(FILES_LABELS_IMG)
        print(FILES_ORIG_IMG)


    



# #\\wsl$\Ubuntu-18.04\home\capilab\proj\opencv_asagau