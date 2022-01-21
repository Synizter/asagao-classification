import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

nbr_of_tile = 0
nbr_plot_row = 3
nbr_plot_col = 4
nn = 1
# #input, cropped image
def tiled_image(crop, patch_size = 200, dest = "."):
    global nn
    for i in range(0,crop.shape[0],patch_size):
        for j in range(0,crop.shape[1],patch_size):
            cc_data = crop[i:i + patch_size, j:j + patch_size, :] # all ch
            if cc_data.shape[0] < patch_size: #patch heigth
                padding = patch_size - cc_data.shape[0]
                patch = np.zeros((padding ,cc_data.shape[1],3))
                print("Heigth Stack",cc_data.shape, patch.shape, nn)    
                cc_data = np.concatenate((cc_data, patch), axis=0)
                print("size after pad", cc_data.shape)
            if cc_data.shape[1] < patch_size: #patch width
                padding = patch_size - cc_data.shape[1]
                patch = np.zeros((cc_data.shape[0] ,padding, 3))
                print("Width Stack",cc_data.shape, patch.shape, nn)  
                cc_data = np.concatenate((cc_data, patch), axis=1)
                print("size after pad", cc_data.shape)
        
            # plt.subplot(nbr_plot_row, nbr_plot_col, nn),plt.imshow(cc_data)
            if(nn>99):
                cv2.imwrite(dest + "/Tile_%d.png" % nn , cc_data)   
            elif(nn > 9):
                cv2.imwrite(dest + "/Tile_0%d.png" % nn , cc_data) 
            else:
                cv2.imwrite(dest + "/Tile_00%d.png" % nn , cc_data)        
            nn += 1
            # cv.imwrite("dataset/validation/asagao_mask_%d.png" % nn, cc_mask)
            # cv.imwrite("dataset/train/asagao_%d.png" % nn, cc_data)
            # print("Dataset patches {} of size {} px saved".format(nn, patch_size))

# if __name__ == "__main__":
#     try:
#     im = cv2.imread('dataset/test/cover1.jpg')
#     print(im.shape)
#     # nbr_of_tile = math.ceil(im.shape[0] * im.shape[1] / (512**2))
#     # print(nbr_of_tile)
#     # tiled_image(im, 512)

#     from glob import glob

#     file = sorted(glob("Tile*"))
#     print(file)

#     temp = np.zeros((len(file) * 512, len(file) * 512, 3))
#     print(temp.shape)

#     col = 0
#     row = 0

#     for i in range(len(file)):
#         #read image
#         tile = cv2.imread(file[i])
#         if(i == 4): #move to next row
#             row += tile.shape[0]
#             col = 0
#         temp[row:row + tile.shape[0], col:col + tile.shape[1],:] = tile
#         col += tile.shape[1]
#         # stride += tile.shape[0]
    

#     cv2.imwrite("test.png", temp[:900, :1600,:])
#     # except Exception as e:
#     # #     print(e)

img = cv2.imread('dataset/test/DJI_20211208095147_0040.JPG')
tiled_image(img, 256, "dataset/test/30m_tiled")