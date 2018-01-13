#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Wang Jun
# @Contact: wangjun@tsingcon.com
# @Date  : 2018/1/10
# @Desc  : 

import os


from area_select import *
from img_pretreat import *
from detect_digital import *


def prepare_imgs():
    i = 1
    for dirpath, dirnames, filenames in os.walk('inputs'):
        for file in filenames:
            fullpath = os.path.join(dirpath, file)
            if fullpath.endswith('jpeg') or fullpath.endswith('jpg') or fullpath.endswith('png'):
                origin_img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
                img = pre_one_img(origin_img)

                cv2.imwrite('thresholds/' + str(file), img)
                plt.figure(i)
                plt.title(str(file))
                print(str(file))
                plt.imshow(img)
                plt.xticks([]), plt.yticks([])
                i += 1

    plt.show()


def cut_img(origin_img, top_left=None, bottom_right=None):
    if top_left is not None and bottom_right is not None:
        cutted_img = origin_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    else:
        cutted_img = origin_img
    return cutted_img


def eval_one(img_in, x_s, x_m, y_s, y_b, y_m, threshold_rate):
    bit0 = np.sum(img_in[0:y_s, x_s:x_s + x_m].flatten())
    bit1 = np.sum(img_in[y_s:y_s + y_b, 0:x_s].flatten())
    bit2 = np.sum(img_in[y_s:y_s + y_b, x_s + x_m:2 * x_s + x_m].flatten())
    bit3 = np.sum(img_in[y_s + y_b:y_s + y_b + y_m, x_s:x_s + x_m].flatten())
    bit4 = np.sum(img_in[y_s + y_b + y_m:y_s + 2 * y_b + y_m, 0:x_s].flatten())
    bit5 = np.sum(img_in[y_s + y_b + y_m:y_s + 2 * y_b + y_m, x_s + x_m:2 * x_s + x_m].flatten())
    bit6 = np.sum(img_in[y_s + 2 * y_b + y_m:2 * y_s + 2 * y_b + y_m, x_s:x_s + x_m].flatten())
    img = img_in.copy()
    cv2.rectangle(img, (x_s, 0), (x_s + x_m, y_s), (0, 255, 255), 2)
    cv2.rectangle(img, (0, y_s), (x_s, y_s + y_b), (0, 255, 255), 2)
    cv2.rectangle(img, (x_s + x_m, y_s), (2 * x_s + x_m, y_s + y_b), (0, 255, 255), 2)
    cv2.rectangle(img, (x_s, y_s + y_b), (x_s + x_m, y_s + y_b + y_m), (0, 255, 255), 2)
    cv2.rectangle(img, (0, y_s + y_b + y_m), (x_s, y_s + 2 * y_b + y_m), (0, 255, 255), 2)
    cv2.rectangle(img, (x_s + x_m, y_s + y_b + y_m), (2 * x_s + x_m, y_s + 2 * y_b + y_m), (0, 255, 255), 2)
    cv2.rectangle(img, (x_s, y_s + 2 * y_b + y_m), (x_s + x_m, 2 * y_s + 2 * y_b + y_m), (0, 255, 255), 2)
    plt.figure(0)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.show()

    print(bit0, bit1, bit2, bit3, bit4, bit5, bit6)
    bit0 = np.sum(img_in[0:y_s, x_s:x_s + x_m].flatten()) < ((y_s * x_m) * 255 * threshold_rate)  # 50000
    bit1 = np.sum(img_in[y_s:y_s + y_b, 0:x_s].flatten()) < ((y_b * x_s) * 255 * threshold_rate)  # 50000
    bit2 = np.sum(img_in[y_s:y_s + y_b, x_s + x_m:2 * x_s + x_m].flatten()) < (
            (y_b * x_s) * 255 * threshold_rate)  # 50000
    bit3 = np.sum(img_in[y_s + y_b:y_s + y_b + y_m, x_s:x_s + x_m].flatten()) < (
            (y_m * x_m) * 255 * threshold_rate)  # 20000
    bit4 = np.sum(img_in[y_s + y_b + y_m:y_s + 2 * y_b + y_m, 0:x_s].flatten()) < (
            (y_b * x_s) * 255 * threshold_rate)  # 50000
    bit5 = np.sum(img_in[y_s + y_b + y_m:y_s + 2 * y_b + y_m, x_s + x_m:2 * x_s + x_m].flatten()) < (
            (y_b * x_s) * 255 * threshold_rate)  # 50000
    bit6 = np.sum(img_in[y_s + 2 * y_b + y_m:2 * y_s + 2 * y_b + y_m, x_s:x_s + x_m].flatten()) < (
            (y_s * x_m) * 255 * threshold_rate)  # 50000
    result = -1
    if bit0:
        # 0 2 3 5 6 7 8 9
        if bit1:
            # 0 5 6 8 9
            if bit2:
                # 0 8 9
                if bit3:
                    # 8 9
                    if bit4:
                        if bit5 and bit6:
                            result = 8
                    else:
                        if bit5 and bit6:
                            result = 9
                else:
                    # 0
                    if bit4 and bit5 and bit6:
                        result = 0
            else:
                # 5 6
                if bit4:
                    if bit3 and bit5 and bit6:
                        result = 5
                else:
                    if bit3 and bit5 and bit6:
                        result = 6

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

    print(result)
    return result


def dmr_single(origin_file, digis_x1=325, digis_y1=340, digis_x2=685, digis_y2=440, digi_w=45, x_s=13, x_m=19, y_m=9,
               y_s=18, y_b=27, threshold_rate=0.7):
    origin_img = cv2.imread(origin_file, cv2.IMREAD_COLOR)
    img = pre_one_img(origin_file)
    cv2.imwrite('threshold.jpg', img)
    # cv2.rectangle(img, (digis_x1, digis_y1), (digis_x2, digis_y2), (0, 255, 255), 2)

    for i in range(int((digis_x2 - digis_x1) / digi_w)):
        digi_img = img[digis_y1:digis_y2, digis_x1 + i * digi_w:digis_x1 + (i + 1) * digi_w]
        ret = eval_one(digi_img, x_s, x_m, y_s, y_b, y_m, threshold_rate)
        if ret >= 0:
            cv2.rectangle(origin_img, (digis_x1 + i * digi_w, digis_y1), (digis_x1 + (i + 1) * digi_w, digis_y2),
                          (255, 255, 0), 2)
            cv2.putText(origin_img, str(ret), (digis_x1 + i * digi_w, digis_y1), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 0), 2)

    cv2.imwrite('output.jpg', origin_img)
    plt.figure(0)
    plt.imshow(origin_img)
    plt.xticks([]), plt.yticks([])
    plt.show()


def test_detect_one_img():
    filename = 'WechatIMG257.jpeg'
    input_file = 'inputs/' + filename
    origin_img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    # top_left = (325, 340)
    # bottom_right = (685, 440)
    # origin_img = cut_img(origin_img, top_left, bottom_right)

    # filename = 'WechatIMG2189-1.jpeg'
    # input_file = 'inputs/' + filename
    # origin_img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    numbers_areas, top_left, bottom_right = detect_one_img(origin_img)

    for numbers_area in numbers_areas:
        x, y, w, h = numbers_area
        origin_img = cv2.rectangle(origin_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    origin_img = cv2.rectangle(origin_img, top_left, bottom_right, (0, 255, 255), 2)
    # epsilon = 0.1 * cv2.arcLength(cnt, True)
    # approx = cv2.approxPolyDP(cnt, epsilon, True)
    # bin_img = cv2.drawContours(origin_img, contours, -1, (0, 255, 0), 3)
    plt.figure(1)
    plt.imshow(origin_img)
    plt.xticks([]), plt.yticks([])
    plt.show()
    cv2.imwrite('outputs/' + filename, origin_img)


def detect_imgs():
    i = 1
    for dirpath, dirnames, filenames in os.walk('inputs'):
        for file in filenames:
            fullpath = os.path.join(dirpath, file)
            if fullpath.endswith('jpeg') or fullpath.endswith('jpg') or fullpath.endswith('png'):
                origin_img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
                numbers_areas, top_left, bottom_right = detect_one_img(origin_img)

                for numbers_area in numbers_areas:
                    x, y, w, h, n = numbers_area
                    origin_img = cv2.rectangle(origin_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(origin_img, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (255, 255, 0), 2)
                origin_img = cv2.rectangle(origin_img, top_left, bottom_right, (0, 255, 255), 2)
                # plt.figure(i)
                # plt.imshow(origin_img)
                # plt.xticks([]), plt.yticks([])
                cv2.imwrite('outputs/' + str(file), origin_img)
                i += 1

    # plt.show()


if __name__ == '__main__':
    train_cnn(save=True)
    # detect_imgs()

