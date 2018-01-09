#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Wang Jun
# @Contact: wangjun@tsingcon.com
# @Date  : 2017/12/30
# @Desc  :
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os


def prepare_img():
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


def pre_one_img(img_in):
    threshold_img = img_in.copy()
    threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 1), np.uint8)
    threshold_img = cv2.bilateralFilter(threshold_img, 9, 75, 75)

    threshold_img = cv2.adaptiveThreshold(threshold_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

    threshold_img = cv2.erode(threshold_img, kernel, iterations=4)
    threshold_img = cv2.dilate(threshold_img, kernel, iterations=2)
    threshold_img = 255 - threshold_img
    return threshold_img


def check_digital(digital_img, origin_img=None, threshold_rate=0.45):
    h, w = digital_img.shape

    if 5 * w < h < 15 * w:
        # 判断是否是1
        bit2 = np.sum(digital_img[0:int(0.5 * h), :].flatten()) > ((w * 0.5 * h) * 255 * threshold_rate)
        bit5 = np.sum(digital_img[int(0.5 * h):h, :].flatten()) > ((w * 0.5 * h) * 255 * threshold_rate)
        if bit2 and bit5:
            # plt.figure(0)
            # plt.imshow(origin_img)
            # plt.xticks([]), plt.yticks([])
            # plt.show()
            result = 1
        else:
            result = -1
    elif 1.5 * w < h < 3 * w:
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
        else:
            result = -1
    else:
        result = -1
    print(result)
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
    # prepare_img()
    detect_imgs()
    # i = 1
    # for dirpath, dirnames, filenames in os.walk('inputs'):
    #     for file in filenames:
    #         fullpath = os.path.join(dirpath, file)
    #         if fullpath.endswith('jpeg') or fullpath.endswith('jpg') or fullpath.endswith('png'):
    #             origin_img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
    #             gray_img = cv2.cvtColor(origin_img, )
    #             plt.figure(i)
    #             plt.imshow(origin_img)
    #             plt.xticks([]), plt.yticks([])
    #             cv2.imwrite('outputs/' + str(file), origin_img)
    #             i += 1
    #
    # plt.show()
