#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Wang Jun
# @Contact: wangjun@tsingcon.com
# @Date  : 2018/1/10
# @Desc  : 
import cv2
import cmd
import sys
import matplotlib.pyplot as plt
import json

from detect_digital import detect_one_img


class AreaSelect(cmd.Cmd):
    def __init__(self, img):
        cmd.Cmd.__init__(self)
        self.do_help(None)
        self.img = img
        numbers_areas, top_left, bottom_right = detect_one_img(origin_img)
        self.area = {'x': top_left[0], 'y': top_left[1], 'w': bottom_right[0] - top_left[0],
                     'h': bottom_right[1] - top_left[1], 'r': 0}

    # 各个命令
    def do_help(self, arg):
        print("-----------------------")
        print("使用说明:")
        print("q: 退出")
        print("s: 显示图片选中区域")
        print("r [int]: 修改旋转角度")
        print("x [int]: 修改区域左上角x坐标")
        print("y [int]: 修改区域左上角y坐标")
        print("w [int]: 修改区域宽度")
        print("h [int]: 修改区域高度")
        print("-----------------------")

    def preloop(self):
        print("请输入命令：")

    def do_quit(self, arg):
        print(str(self.area))
        print("退出区域选择!")
        with open('area.ini', 'w') as f:
            f.write(json.dumps(self.area))
        sys.exit(1)

    def do_show(self, arg):
        print(str(self.area))
        img = self.img.copy()
        rows = img.shape[0]
        cols = img.shape[1]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), self.area['r'], 1)
        img = cv2.warpAffine(img, M, (cols, rows))
        img = cv2.rectangle(img, (self.area['x'], self.area['y']),
                            (self.area['x'] + self.area['w'], self.area['y'] + self.area['h']), (0, 255, 255), 2)
        w = 0
        while w < self.area['w']:
            img = cv2.putText(img, str(self.area['x'] + w), (self.area['x'] + w, self.area['y']),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            w += 50

        h = 0
        while h < self.area['h']:
            img = cv2.putText(img, str(self.area['y'] + h), (self.area['x'] - 40, self.area['y'] + h),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            h += 50

        plt.figure(1)
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])
        plt.show()

    def do_x(self, arg):
        self.area['x'] = int(arg)

    def do_y(self, arg):
        self.area['y'] = int(arg)

    def do_w(self, arg):
        self.area['w'] = int(arg)

    def do_h(self, arg):
        self.area['h'] = int(arg)

    def do_r(self, arg):
        self.area['r'] = int(arg)

    # 快捷键
    do_q = do_quit
    do_s = do_show


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        filename = '1.2-晚8点燃气表.jpg'  # 'calculator01.jpg'
        input_file = 'data/inputs/' + filename
    origin_img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    area_select = AreaSelect(origin_img)
    area_select.cmdloop()
