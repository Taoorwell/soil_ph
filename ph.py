import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from glob import glob
from matplotlib import pyplot as plt
from skimage import io


def create_datasets_from_sample_ph(ph_path):
    '''
    :param ph_path: fold path contained standard ph sample images
    :return: an array with shape (62500*7, 4), including rbg values and corresponding ph label.
    '''
    ph_files = os.listdir(ph_path)
    datasets = np.zeros((62500*7, 4), dtype=np.float32)
    for i, ph in enumerate(ph_files):
        ph_v = float(ph.split('.')[0] + str('.') + ph.split('.')[1])
        rgb = io.imread(ph_path + ph)
        rgb = rgb[400:650, 550:800]
        rgb = np.reshape(rgb, (-1, 3)) / 255
        # rgb = np.reshape(rgb, (-1, 3))
        ph_v = np.ones((rgb.shape[0], 1), dtype=np.float32) * ph_v
        data = np.concatenate((rgb, ph_v), axis=1)
        datasets[i*62500:(i+1)*62500] = data
    return datasets

# # plot data # #
# plt.figure(figsize=(12, 9))
# plt.subplot(131)
# plt.scatter(test_datasets[:, 0], test_datasets[:, -1], c='r')
# plt.ylabel('pH')
# plt.xlabel('R')
#
# plt.subplot(132)
# plt.scatter(test_datasets[:, 1], test_datasets[:, -1], c='g')
# plt.ylabel('pH')
# plt.xlabel('G')
#
# plt.subplot(133)
# plt.scatter(test_datasets[:, 2], test_datasets[:, -1], c='b')
# plt.ylabel('pH')
# plt.xlabel('B')
#
# plt.show()

# # curve fitting # #


# def func(x, a, b, c, d):
#     R, G, B = x[:, 0], x[:, 1], x[:, 2]
#     return a*R + b*G + c*B + d
#
#
# popt, _ = curve_fit(func, train_datasets[:, :3], train_datasets[:, -1])
# a, b, c, d = popt
# print(a, b, c, d)
# # -5.942919252287673 5.9305925444836225 -0.9876440954361952 7.111179588962335
# test_pre = func(test_datasets[:, :3], a=a, b=b, c=c, d=d)
# mse = mean_squared_error(test_datasets[:, -1], test_pre)
# print(mse)
# # 0.078113504

def model_train_saving(datasets):
    '''
    :param datasets: datasets manipulated through previous operation
    :return: trained nn model, name end with .h5, for further use
    '''
    # tran and test tensorflow datasets preparation
    def parse(x):
        return x[:3], x[-1]
    train_datasets, test_datasets = train_test_split(datasets, test_size=0.1, shuffle=True)
    train_datasets = tf.data.Dataset.from_tensor_slices(train_datasets)
    train_datasets = train_datasets.map(parse).batch(batch_size=50).repeat()

    test_datasets = tf.data.Dataset.from_tensor_slices(test_datasets)
    test_datasets = test_datasets.map(parse).batch(batch_size=50).repeat()

    # simple nn prediction model construction
    x_in = tf.keras.layers.Input(shape=(3, ))
    x = tf.keras.layers.Dense(10, activation='relu')(x_in)
    x = tf.keras.layers.Dense(5, activation='relu')(x)
    x_out = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(x_in, x_out)

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=[tf.keras.metrics.MSE])
    # model fitting
    model.fit(train_datasets, epochs=20, steps_per_epoch=7875,
              validation_data=test_datasets, validation_steps=875)
    # model saving in current file with name end with .h5
    model.save('model.h5')
    return model


if __name__ == '__main__':
    # some parameters
    ph_path = r'pH-calibration/'
    figures_path = r'../Figures/**/**/**/'
    if os.path.exists(r'model.h5'):
        print('found trained model in current folder, starting to loading existed model')
        model = tf.keras.models.load_model('model.h5')
    else:
        print('model file unfounded, starting to make datasets and train model from scratch')
        datasets = create_datasets_from_sample_ph(ph_path)
        model = model_train_saving(datasets)
        print('model train finish, save in current folder named model.h5')

    # Read ph image files and crop for prediction # # #
    print('starting to make prediction')
    images = glob(figures_path + '*.png', recursive=True)
    for image_path in tqdm(images):
        # print(image_path)
        # load image
        image = io.imread(image_path)[150:900, 200:900]
        # crop image background
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        t = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
        x, y, w, h = cv.boundingRect(t[1])
        image = image[y:y + h, x:x + w]
        # flip image
        image = np.flipud(image)
        # crop image again
        image = image[50+70:-5, 20:-50]
        a1, b1, _ = image.shape

        # reshape image for prediction
        image_ = np.float32(np.reshape(image, (-1, 3)) / 255)
        # model prediction
        ph_p = model.predict(image_)
        ph_p = np.reshape(ph_p, (a1, b1))

        plt.ioff()
        fig = plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(image)
        plt.xlabel('raw image')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(122)
        plt.imshow(ph_p, cmap='RdYlGn', vmin=5.4, vmax=7.6)
        plt.colorbar()
        plt.xlabel('pH_predicted')
        plt.xticks([])
        plt.yticks([])
        # plt.close()
        plt.savefig(os.path.abspath(image_path).split('.')[-2] + '_pre.svg')
        plt.close(fig)
        plt.imsave(os.path.abspath(image_path).split('.')[-2] + '_pre.tiff', ph_p)
        # plt.show()
        # break
    # print(image_path, 'save successful')


# make prediction using nn model
# model = tf.keras.models.load_model('model.h5')

# make prediction using multi variables regression
# ph_p = func(ph_000011, a=a, b=b, c=c, d=d)
# s = np.reshape(ph_p, (a1, b1))

