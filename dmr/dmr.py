#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Wang Jun
# @Contact: wangjun@tsingcon.com
# @Date  : 2017/12/30
# @license : Copyright(C), Beijing Tsing Con Technology Co., Ltd.
# @Desc  : 
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os


def match_digital(img_in):
    img = img_in.copy()
    method = cv2.TM_SQDIFF
    match_res = []
    for i in range(10):
        pts = []
        for dirpath, dirnames, filenames in os.walk('templates/' + str(i)):
            for file in filenames:
                fullpath = os.path.join(dirpath, file)
                template = cv2.imread(fullpath, 0)
                W, H = img.shape[::-1]
                w, h = template.shape[::-1]
                fx = W / w / 8
                fy = H / h
                not_matched = True
                template_resized = cv2.resize(template, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(img, template_resized, method)
                threshold = 900000
                loc = np.where(res <= threshold)
                print('i:', i, 'file:', file, 'min:', min(res.flatten()))
                for pt in zip(*loc[::-1]):
                    pts.append({'top_left': pt, 'bottom_right': (int(pt[0] + w * fx), int(pt[1] + h * fy))})
                    not_matched = False
                if not not_matched:
                    break
            break
        match_res.append(pts)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        #     top_left = min_loc
        # else:
        #     top_left = max_loc
        # bottom_right = (top_left[0] + w, top_left[1] + h)
        # return top_left, bottom_right, i
    return match_res


# def match_digital(img_in):
#     img = img_in.copy()
#     method = cv2.TM_SQDIFF
#     match_res = []
#     for i in range(10):
#         pts = []
#         for dirpath, dirnames, filenames in os.walk('templates/' + str(i)):
#             for file in filenames:
#                 fullpath = os.path.join(dirpath, file)
#                 template = cv2.imread(fullpath, 0)
#                 W, H = img.shape[::-1]
#                 w, h = template.shape[::-1]
#                 fact = 0.5 * H / h
#                 not_matched = True
#                 while fact > 0.01 and not_matched:
#                     template_resized = cv2.resize(template, None, fx=fact, fy=fact, interpolation=cv2.INTER_AREA)
#                     res = cv2.matchTemplate(img, template_resized, method)
#                     threshold = 900000
#                     loc = np.where(res <= threshold)
#                     print('i:', i, 'file:', file, 'h:', int(h*fact), 'min:', min(res.flatten()))
#                     for pt in zip(*loc[::-1]):
#                         pts.append({'top_left': pt, 'bottom_right': (int(pt[0] + w * fact), int(pt[1] + h * fact))})
#                         not_matched = False
#                     fact -= 0.001
#                 if not not_matched:
#                     break
#             break
#         match_res.append(pts)
#     return match_res


def prepare_img():
    i = 1
    for dirpath, dirnames, filenames in os.walk('inputs'):
        for file in filenames:
            fullpath = os.path.join(dirpath, file)
            img = pre_one_img(fullpath)

            cv2.imwrite('outputs/' + str(file), img)
            plt.figure(i)
            plt.title(str(file))
            print(str(file))
            plt.imshow(img)
            plt.xticks([]), plt.yticks([])
            i += 1

    plt.show()


def pre_one_img(origin_file):
    threshold_img = cv2.imread(origin_file, 0)
    kernel = np.ones((2, 2), np.uint8)
    threshold_img = cv2.bilateralFilter(threshold_img, 9, 75, 75)

    threshold_img = cv2.adaptiveThreshold(threshold_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

    threshold_img = cv2.erode(threshold_img, kernel, iterations=1)
    return threshold_img


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


if __name__ == '__main__':
    # input_file = 'inputs/WechatIMG257.jpeg'
    # dmr_single(input_file, digis_x1=325, digis_y1=340, digis_x2=685, digis_y2=440, digi_w=45, x_s=13, x_m=19, y_m=9,
    #            y_s=18, y_b=27)
    input_file = 'inputs/WechatIMG257.jpeg'
    origin_img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    # pt1 = np.float32([[384, 290], [880, 290], [404, 430], [880, 430]])
    # pt2 = np.float32([[0, 0], [476, 0], [0, 140], [476, 140]])
    # matrix = cv2.getPerspectiveTransform(pt1, pt2)
    # warp_img = cv2.warpPerspective(origin_img, matrix, (476, 140))
    # cv2.imwrite('inputs/WechatIMG2189-1.jpeg', warp_img)
    # cv2.rectangle(origin_img, (404, 290), (880, 430), (0, 255, 255), 2)
    gray = pre_one_img(input_file)
    orb = cv2.ORB.create()
    kp, des = orb.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray, kp, origin_img, color=(0, 255, 0), flags=0)

    plt.figure(0)
    plt.imshow(origin_img)
    plt.xticks([]), plt.yticks([])
    plt.show()
    # input_file = 'inputs/WechatIMG2189-1.jpeg'
    # dmr_single(input_file, digis_x1=0, digis_y1=0, digis_x2=476, digis_y2=140, digi_w=68, x_s=20, x_m=28, y_m=13,
    #            y_s=25, y_b=38)
