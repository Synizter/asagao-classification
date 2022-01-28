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

#change to 'if 1' to load and create training/testing set
if 1:
    model = get_unet_model(256,256,3)
    # opt = Adam(learning_rate = 0.001)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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
    import time

    tic = time.time()
    print("Start loading dataset .....")
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
    print("Done loading {} data in {}s.".format(asagao_dataset.shape[0], time.time() - tic))

if 0:

    history = model.fit(x=aug.flow(X_train, y_train, batch_size=8),
                        verbose = 1,
                        batch_size = 16,
                        epochs = 100,
                        validation_data = (X_test, y_test),
                        shuffle = False)
    model.save("50m_30mUnet_All")


    import matplotlib.pyplot as plt

    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training acc',linewidth=2)
    plt.plot(epochs, val_acc, 'r--', label='Validation acc',linewidth=2)
    plt.title('Training  accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(epochs, loss, 'b', label='Training loss')

    plt.plot(epochs, val_loss, 'k', label='Validationloss ')
    plt.title('Training loss')
    plt.legend()

    plt.show()

model = tf.keras.models.load_model('50m_30mUnet_All')
# loss, acc = model.evaluate(X_test, y_test, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# print(model.predict(X_test).shape)

# tile the test images
def predict_tile(tile, sens = 0.001):
    tile = tile.astype('float32') / 255.0
    test_img_other_input=np.expand_dims(tile, 0)
    prob_map = (model.predict(test_img_other_input) > sensitivty).astype(np.uint8)
    return prob_map[0]
    