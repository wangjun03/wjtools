#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Wang Jun
# @Contact: wangjun@tsingcon.com
# @Date  : 2018/1/10
# @Desc  :
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def plt_img(figure_index, img):
    plt.figure(figure_index)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.show()


def img2bin(img_in, erode_param=None, show_img=False):
    threshold_img = img_in.copy()
    threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_BGR2GRAY)

    threshold_img = cv2.bilateralFilter(threshold_img, 7, 75, 75)

    threshold_img = cv2.adaptiveThreshold(threshold_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    if erode_param is not None:
        kernel = np.ones((erode_param[0], erode_param[1]), np.uint8)
        threshold_img = cv2.erode(threshold_img, kernel, iterations=erode_param[2])
        threshold_img = cv2.dilate(threshold_img, kernel, iterations=erode_param[3])
    if show_img:
        plt_img(0, threshold_img)

    return threshold_img

