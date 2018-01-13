#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Wang Jun
# @Contact: wangjun@tsingcon.com
# @Date  : 2018/1/10
# @Desc  :
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import os
import uuid

from img_pretreat import *


def check_digital(digital_img, origin_img=None, threshold_rate=0.45):
    h, w = digital_img.shape

    # if 3 * w < h < 15 * w:
    # 判断是否是1
    # bit2 = np.sum(digital_img[0:int(0.5 * h), :].flatten()) > ((w * 0.5 * h) * 255 * threshold_rate)
    # bit5 = np.sum(digital_img[int(0.5 * h):h, :].flatten()) > ((w * 0.5 * h) * 255 * threshold_rate)
    # if bit2 and bit5:
    # plt.figure(0)
    # plt.imshow(origin_img)
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    # result = 1
    # else:
    #     result = -1
    # elif 1.5 * w < h < 3 * w:
    if 1.2 * w < h < 15 * w:
        x_s = int(0.25 * w)
        x_m = int(0.5 * w)
        y_m = int(0.125 * h)
        y_s = int(0.125 * h)
        y_b = int(0.3 * h)

        bit0 = np.sum(digital_img[0:y_s, x_s:x_s + x_m].flatten()) > ((y_s * x_m) * 255 * threshold_rate)
        bit1 = np.sum(digital_img[y_s:y_s + y_b, 0:x_s].flatten()) > ((y_b * x_s) * 255 * threshold_rate)
        bit2 = np.sum(digital_img[y_s:y_s + y_b, x_s + x_m:2 * x_s + x_m].flatten()) > (
                (y_b * x_s) * 255 * threshold_rate)
        bit3 = np.sum(digital_img[y_s + y_b:y_s + y_b + y_m, x_s:x_s + x_m].flatten()) > (
                (y_m * x_m) * 255 * threshold_rate)
        bit4 = np.sum(digital_img[y_s + y_b + y_m:y_s + 2 * y_b + y_m, 0:x_s].flatten()) > (
                (y_b * x_s) * 255 * threshold_rate)
        bit5 = np.sum(digital_img[y_s + y_b + y_m:y_s + 2 * y_b + y_m, x_s + x_m:2 * x_s + x_m].flatten()) > (
                (y_b * x_s) * 255 * threshold_rate)
        bit6 = np.sum(digital_img[y_s + 2 * y_b + y_m:2 * y_s + 2 * y_b + y_m, x_s:x_s + x_m].flatten()) > (
                (y_s * x_m) * 255 * threshold_rate)
        emply0 = np.sum(digital_img[y_s:y_s + y_b, x_s:x_s + x_m].flatten()) > (
                (y_b * x_m) * 255 * threshold_rate)
        emply1 = np.sum(digital_img[y_s + y_b + y_m:y_s + 2 * y_b + y_m, x_s:x_s + x_m].flatten()) > (
                (y_b * x_m) * 255 * threshold_rate)
        result = -1
        if not (emply0 and emply1):
            if bit0:
                # 0 2 3 5 6 7 8 9
                if bit1:
                    # 0 5 6 7 8 9
                    if bit2:
                        # 0 7 8 9
                        if bit3:
                            # 8 9
                            if bit4:
                                if bit5 and bit6:
                                    result = 8
                            else:
                                if bit5 and bit6:
                                    result = 9
                        else:
                            # 0 7
                            if bit4:
                                if bit5 and bit6:
                                    result = 0
                            else:
                                if bit5 and not bit6:
                                    result = 7
                    else:
                        # 5 6
                        if bit4:
                            if bit3 and bit5 and bit6:
                                result = 6
                        else:
                            if bit3 and bit5 and bit6:
                                result = 5

                else:
                    # 2 3 7
                    if bit3:
                        # 2 3
                        if bit4:
                            if bit2 and not bit5 and bit6:
                                result = 2
                        else:
                            if bit2 and bit5 and bit6:
                                result = 3
                    else:
                        if bit2 and not bit4 and bit5 and not bit6:
                            result = 7
            else:
                # 1 4
                if bit1:
                    # 4
                    if bit2 and bit3 and not bit4 and bit5 and not bit6:
                        result = 4
                else:
                    # 1
                    if bit2 and not bit3 and not bit4 and bit5 and not bit6:
                        result = 1
        else:
            result = -1
    else:
        result = -1
    # print(result)
    return result


def detect_one_img(img_in):
    threshold_img = pre_one_img(img_in)
    o_h = threshold_img.shape[0]
    o_w = threshold_img.shape[1]
    bin_img, contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_x = 10000
    min_y = 10000
    max_x = 0
    max_y = 0
    numbers_areas = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 0.3 * o_h > h > 0.03 * o_h and 0.1 * o_h < y < 0.45 * o_h and 0.2 * o_w < x < 0.7 * o_w:
            res = check_digital(threshold_img[y:y + h, x:x + w], img_in[y:y + h, x:x + w])
            if res > -1:
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x + w > max_x:
                    max_x = x + w
                if y + h > max_y:
                    max_y = y + h
                numbers_areas.append((x, y, w, h, res))
    return numbers_areas, (min_x, min_y), (max_x, max_y)


def detect_cutted_img(img_in):
    threshold_img = img2bin(img_in)
    o_h = threshold_img.shape[0]
    o_w = threshold_img.shape[1]
    bin_img, contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_x = 10000
    min_y = 10000
    max_x = 0
    max_y = 0
    numbers_areas = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if y < 0.2 * o_h:  # o_h > h > 0.5 * o_h: #and 0.1 * h < w < 0.4 * h:
            # res = check_digital(threshold_img[y:y + o_h, x:x + w], img_in[y:y + h, x:x + w])
            res = check_digital(threshold_img[y:y + o_h, x:x + w], img_in[y:y + h, x:x + w], threshold_rate=0.2)
            if res > -1:
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x + w > max_x:
                    max_x = x + w
                if y + h > max_y:
                    max_y = y + h
                numbers_areas.append((x, y, w, h, res))
    return numbers_areas, (min_x, min_y), (max_x, max_y), contours


def split_digital(img_in, erode_param=None, show_img=False):
    bin_img = img2bin(img_in, erode_param, show_img)

    o_h = bin_img.shape[0]
    o_w = bin_img.shape[1]
    bin_img = 255 - bin_img
    bin_img, contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    numbers_areas = []
    scan_x = 0
    while scan_x < o_w:
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if x <= scan_x <= x + w:
                h = o_h - y
                scan_x = x + w - 10
                if h > 0.6 * o_h and 1.4 * h >= w >= 0.1 * h:
                    digital_img = bin_img[y:y + h, x:x + w]
                    numbers_areas.append((x, y, w, h, digital_img))
                del contours[i]
                break
        scan_x += 1

    # for cnt in contours:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     if y < 0.2 * o_h:
    #         h = o_h - y
    #         numbers_areas.append((x, y, w, h))
    return numbers_areas


def dig_train_data(file_path='data/train_data/'):
    x_train = []
    y_train = []
    i = 0
    for i in range(11):
        for dirpath, dirnames, filenames in os.walk(file_path + str(i)):
            for file in filenames:
                fullpath = os.path.join(dirpath, file)
                if fullpath.endswith('jpeg') or fullpath.endswith('jpg') or fullpath.endswith('png'):
                    template = cv2.imread(fullpath, 0)
                    x_train.append(template)
                    y_train.append(i)
    return np.array(x_train), np.array(y_train), i + 1


def train_cnn(save=False, file_path='data/train_data/', batch_size=32, epochs=50):
    x_train, y_train, num_classes = dig_train_data(file_path)
    # input image dimensions
    img_rows, img_cols = x_train[0].shape
    # the data, shuffled and split between train and test sets
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_train /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    model = Sequential()
    model.add(Conv2D(32,
                     activation='relu',
                     input_shape=input_shape,
                     nb_row=3,
                     nb_col=3))
    model.add(Conv2D(64, activation='relu',
                     nb_row=3,
                     nb_col=3))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.metrics.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1)
    if save:
        json_string = model.to_json()
        with open(file_path + 'cnn_arch.json', 'w') as f:
            f.write(json_string)
            model.save_weights(file_path + 'cnn_w.h5')
            print('saved to', file_path)


# def train_cnn(save=False, file_path=None, batch_size=128, num_classes=10, epochs=12):
#     # input image dimensions
#     img_rows, img_cols = 28, 28
#     # the data, shuffled and split between train and test sets
#     # with open(, 'rb') as f:
#     (x_train, y_train), (x_test, y_test) = mnist.load_data(os.getcwd() + "/data/train_data/mnist.npz")
#     if K.image_data_format() == 'channels_first':
#         x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#         x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#         input_shape = (1, img_rows, img_cols)
#     else:
#         x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#         x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#         input_shape = (img_rows, img_cols, 1)
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#     x_train /= 255
#     x_test /= 255
#     print('x_train shape:', x_train.shape)
#     print(x_train.shape[0], 'train samples')
#     print(x_test.shape[0], 'test samples')
#     # convert class vectors to binary class matrices
#     y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
#     y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
#     model = Sequential()
#     model.add(Conv2D(32,
#                      activation='relu',
#                      input_shape=input_shape,
#                      nb_row=3,
#                      nb_col=3))
#     model.add(Conv2D(64, activation='relu',
#                      nb_row=3,
#                      nb_col=3))
#     # model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.35))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes, activation='softmax'))
#     model.compile(loss=keras.metrics.categorical_crossentropy,
#                   optimizer=keras.optimizers.Adadelta(),
#                   metrics=['accuracy'])
#     model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
#               verbose=1, validation_data=(x_test, y_test))
#     score = model.evaluate(x_test, y_test, verbose=0)
#     if save:
#         json_string = model.to_json()
#         if file_path is None:
#             file_path = os.getcwd() + "/data/cnn_model/mnist"
#         open(file_path + '_arch.json', 'w').write(json_string)
#         model.save_weights(file_path + '_w.h5')
#     print('Test loss:', score[0])
#     print('Test accuracy:', score[1])


def load_cnn(file_path=None):
    if file_path is None:
        file_path = os.getcwd() + "/data/cnn_model/mnist"
    model = model_from_json(open(file_path + '_arch.json').read())
    model.load_weights(file_path + '_w.h5')
    model.compile(loss=keras.metrics.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def dig_predict(dig_img, cnn_model, img_rows=28, img_cols=28, batch_size=32, save_img=False):
    fit_img = np.zeros((img_rows, img_cols), np.uint8)
    h, w = dig_img.shape
    fit_h = img_rows - 8
    fit_w = int((img_rows - 8) * w / h)
    if fit_h <= 0 or fit_w <= 0:
        return -1
    dig_img = cv2.resize(dig_img, (fit_w, fit_h), interpolation=cv2.INTER_CUBIC)
    fit_img[int((img_rows - fit_h) / 2):int((img_rows + fit_h) / 2),
    int((img_cols - fit_w) / 2):int((img_cols + fit_w) / 2)] = dig_img
    fit_img = cv2.bilateralFilter(fit_img, 7, 75, 75)

    res, fit_img = cv2.threshold(fit_img, 200, 255, cv2.THRESH_BINARY)

    x_img = fit_img.copy()

    x_img = x_img.reshape(1, img_rows, img_cols, 1)
    x_img = x_img.astype('float32')
    x_img /= 255
    res = cnn_model.predict(x_img, batch_size=batch_size)
    res_li = res.flatten().tolist()
    n = res_li.index(max(res_li))
    if n == 10:
        n = -1
        if save_img:
            cv2.imwrite('data/train_data/' + str(uuid.uuid1()) + '.jpg', fit_img)
    # print(res_li)
    # print(n)
    return n


def cut_area(img, area):
    img = img.copy()
    rows = img.shape[0]
    cols = img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), area['r'], 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    cutted_img = img[area['y']:area['y'] + area['h'], area['x']:area['x'] + area['w']]
    return cutted_img, img


def dmr_single(img_in, area, model, img_rows=28, img_cols=28, erode_param=None, show_img=False):
    cutted_img, origin_img = cut_area(img_in, area)
    numbers_areas = split_digital(cutted_img, erode_param, show_img)

    x_res = []
    for numbers_area in numbers_areas:
        x, y, w, h, dig_img = numbers_area
        n = dig_predict(dig_img, model, img_rows, img_cols)
        if show_img:
            origin_img = cv2.rectangle(origin_img, (area['x'] + x, area['y'] + y),
                                       (area['x'] + x + w, area['y'] + y + h), (0, 255, 0), 2)
            cv2.putText(origin_img, str(n), (area['x'] + x, area['y'] + y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2)
        x_res.append((x, n))
    if show_img:
        plt.figure(0)
        plt.imshow(origin_img)
        plt.xticks([]), plt.yticks([])
        plt.show()
    ordered_res = sorted(x_res, key=lambda x_value: x_value[0])
    res = [x_value[1] for x_value in ordered_res]
    return res


def combine_digits(digits):
    result = -1
    for digital in digits:
        if 0 <= digital <= 9:
            if result == -1:
                result = digital
            else:
                result = result * 10 + digital
        else:
            if result == -1:
                continue
            else:
                result = -1
                break
    # print(result)
    return result


def dmr_number(origin_img, area, model, erode_param):
    digits = dmr_single(origin_img, area, model, erode_param=erode_param, show_img=False)
    res = combine_digits(digits)
    if res < 0:
        cutted_img, temp_img = cut_area(origin_img, area)
        cv2.imwrite('data/' + str(uuid.uuid1()) + '.jpg', cutted_img)
    return res


if __name__ == '__main__':
    train_cnn(save=True)
