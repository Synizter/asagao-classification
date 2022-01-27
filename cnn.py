# import tensorflow as tf
# from glob import glob
# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split

# import os
# from tiled_detiled import tiled_image
# import time
# import math

# #read dataset
# img_dataset = []
# msk_dataset = []

# f_img = sorted(glob('dataset/train/asagao/*'))
# for f in f_img:
#     image = cv2.imread(f)
#     image = image.astype('float32') / 255.0
#     img_dataset.append(np.array(image))

# f_msk = sorted(glob('dataset/validation/asagao/*'))
# for f in f_msk:
#     image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
#     (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]

#     msk_dataset.append(np.array(image))


# asagao_dataset = np.array(img_dataset)
# # #D not normalize masks, just rescale to 0 to 1.a
# asagao_msk_dataset = np.expand_dims(np.array(msk_dataset) /255., -1)

# X_train, X_test, y_train, y_test = train_test_split(asagao_dataset, asagao_msk_dataset, test_size= 0.10, random_state=0)
# # y_train = to_categorical(y_train, 3)

# # #define model
# # import tensorflow as tf
# # from tensorflow.keras.models import Model
# # from tensorflow.keras.layers import Dense, Dropout, Input
# # from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout,Conv2DTranspose, concatenate
# # from tensorflow.keras.optimizers import Adam
    
# # # model.summary()
# # input_w = 256
# # input_h = 256
# # input_ch = 3

# # s  = Input((input_w, input_h, input_ch))
# # #Downscale path -----------------------------------------------------------------------------------
# # c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
# # c1 = Dropout(0.1)(c1)
# # c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
# # p1 = MaxPooling2D((2, 2))(c1)
# # #100
# # c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
# # c2 = Dropout(0.1)(c2)
# # c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
# # p2 = MaxPooling2D((2, 2))(c2)
# # #50
# # c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
# # c3 = Dropout(0.2)(c3)
# # c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
# # p3 = MaxPooling2D((2, 2))(c3)
# # #15
# # c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
# # c4 = Dropout(0.2)(c4)
# # c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
# # p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
# # c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
# # c5 = Dropout(0.3)(c5)
# # c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# # #Scale up path ----------------------------------------------------------------------------------
# # u6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c5)
# # u6 = concatenate([u6, c4])
# # c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
# # c6 = Dropout(0.2)(c6)
# # c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
# # u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
# # u7 = concatenate([u7, c3])
# # c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
# # c7 = Dropout(0.2)(c7)
# # c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
# # u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
# # u8 = concatenate([u8, c2])
# # c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
# # c8 = Dropout(0.1)(c8)
# # c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
# # u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
# # u9 = concatenate([u9, c1], axis=3)
# # c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
# # c9 = Dropout(0.1)(c9)
# # c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
# # outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
# # model = Model(inputs=[s], outputs=[outputs])
# # opt = Adam(learning_rate = 0.001)
# # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
# # model.summary()

# # #train
# # history = model.fit(X_train, 
# #                     y_train,
# #                     verbose = 1,
# #                     batch_size = 16,
                    
# #                     epochs = 100,
# #                     validation_data = (X_test, y_test),
# #                     shuffle = False)
# # model.save("50m_30m_Unet")

# # #plot thte result
# # import matplotlib.pyplot as plt

# # acc = history.history['accuracy']
# # loss = history.history['loss']
# # val_acc = history.history['val_accuracy']
# # val_loss = history.history['val_loss']
# # epochs = range(len(acc))

# # plt.figure(figsize=(15, 6))
# # plt.subplot(1, 2, 1)
# # plt.plot(epochs, acc, 'b', label='Training acc',linewidth=2)
# # plt.plot(epochs, val_acc, 'r--', label='Validation acc',linewidth=2)
# # plt.title('Training  accuracy')
# # plt.legend()

# # plt.subplot(1, 2, 2)

# # plt.plot(epochs, loss, 'b', label='Training loss')

# # plt.plot(epochs, val_loss, 'k', label='Validationloss ')
# # plt.title('Training loss')
# # plt.legend()

# # plt.show()

# model = tf.keras.models.load_model('50m_30m_Unet')
# sensitivity = 0.3


# # loss, acc = model.evaluate(X_test, y_test, verbose=2)
# # print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# target_f = "dataset/raw/not_labeled/30m-1/DJI_20211208095102_0006.JPG"



# def predict_tile(tile, sens = 0.001):
#     tile = tile.astype('float32') / 255.0
#     test_img_other_input=np.expand_dims(tile, 0)
#     prob_map = (model.predict(test_img_other_input) > sens).astype(np.uint8)
#     return prob_map[0]

# if len(os.listdir('dataset/test/test1')) != 0:
#     print("Found previous tiled images, attemp delete")
#     file_to_delete = os.listdir('dataset/test/test1')
#     for f in file_to_delete:
#         os.remove('dataset/test/test1/' + f)

# target_im = cv2.imread(target_f)
# if target_im is not None:
#     tiled_image(target_im, 256, "dataset/test/test1")
# else:
#     print("Cannot read ifle")


# #Slice image
# predict_file = sorted(glob("dataset/test/test1/*.png"))
# predict_file = iter(predict_file)
# # print(predict_file)

# predicted_list = []

# #remove existing file
# if len(os.listdir('dataset/test/result_test1')) != 0:
#     print("Found previous predicted files, attemp delete")
#     file_to_delete = os.listdir('dataset/test/result_test1')
#     for f in file_to_delete:
#         os.remove('dataset/test/result_test1/' + f)
# #predict slided image
# nn = 1
# for f in predict_file:
#     target = f
#     # print(target)
#     inputs = cv2.imread(target)
#     tic = time.time()
#     r = predict_tile(inputs, sensitivity)
#     toc = time.time()
#     print("predicted %s in %.8f s" %(f, toc - tic))
#     tf.keras.preprocessing.image.save_img('dataset/test/result_test1/result_%s.png'% str(nn).zfill(4),r)
#     nn += 1

# #save otuput
# result_heigth = 5460
# result_width = 8192
# tile_size = 256


# file = sorted(glob("dataset/test/result_test1/*"))

# temp = np.zeros((math.ceil(result_heigth / tile_size) * tile_size, math.ceil(result_width / tile_size) * tile_size,3))

# i = 0
# for r in range(0, temp.shape[0], tile_size):
#     for c in range(0, temp.shape[1], tile_size):
#         tile = cv2.imread(file[i])
#         temp[r:r + tile_size, c:c + tile_size,:] = tile 
#         i += 1

# cv2.imwrite("predicted_location.jpg" , temp[:5460, :8192,:])
# # cv2.imwrite("predicted_location_DJI_20211208095147_0040.JPG" , temp[:5460, :8192,:])

import tensorflow as tf
from glob import glob
import os
import cv2
from PIL import Image
from tensorflow.keras.utils import normalize
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from mymodel import get_unet_model
from tensorflow.keras.optimizers import Adam

model = get_unet_model(256,256,3)
opt = Adam(learning_rate = 0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
print(type(model))

aug = ImageDataGenerator(
	rotation_range=180,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
total = 0

from sklearn.model_selection import train_test_split

img_dataset = []
msk_dataset = []


f_img = sorted(glob('dataset/train/asagao/*'))
f_msk = sorted(glob('dataset/validation/asagao/*'))
for img, msk in zip(f_img, f_msk):
    mask = cv2.imread(msk, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(img)

    image = image.astype('float32') / 255.0
    (thresh, im_bw) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)[1]
    if mask.shape[0] != 256 or mask.shape[1] != 256:
        print(msk, 'was invalid size')
        continue
    # image = image.resize((SIZE, SIZE))

    img_dataset.append(np.array(image))
    msk_dataset.append(np.array(mask))


f_img = sorted(glob('dataset/train/not_asagao/*'))
f_msk = sorted(glob('dataset/validation/not_asagao/*'))
print(len(f_img), len(f_msk))
for img, msk in zip(f_img, f_msk):
    mask = cv2.imread(msk, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(img)

    image = image.astype('float32') / 255.0
    (thresh, im_bw) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)[1]
    if mask.shape[0] != 256 or mask.shape[1] != 256:
        print(msk, 'was invalid size')
        continue
    # image = image.resize((SIZE, SIZE))

    img_dataset.append(np.array(image))
    msk_dataset.append(np.array(mask))


asagao_dataset = np.array(img_dataset)
asagao_msk_dataset = np.expand_dims(np.array(msk_dataset) /255., -1)
X_train, X_test, y_train, y_test = train_test_split(asagao_dataset, asagao_msk_dataset, test_size= 0.10, random_state=0)


del f_img, f_msk, image, im_bw, thresh, mask, msk, img

history = model.fit(x=aug.flow(X_train, y_train, batch_size=8),
                    verbose = 1,
                    batch_size = 8,
                    epochs = 100,
                    validation_data = (X_test, y_test),
                    shuffle = False)
model.save("50m_30m_10mUnet_All")