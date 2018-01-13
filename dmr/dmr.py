#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Wang Jun
# @Contact: wangjun@tsingcon.com
# @Date  : 2017/12/30
# @Desc  :
import getopt
import time
# from picamera.array import PiRGBArray
# from picamera import PiCamera

from area_select import *
from detect_digital import *
import logging

# 配置日志信息
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./logs/dmr.log',
                    filemode='a')
# 定义一个Handler打印INFO及以上级别的日志到sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# 设置日志打印格式
formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
# 将定义好的console日志handler添加到root logger
logging.getLogger('').addHandler(console)


def load_area_ini(filename='area.ini'):
    with open(filename, 'r') as f:
        info = f.read()
        area_ini = json.loads(info)
    return area_ini


def print_usage():
    print("使用说明:")
    print("dmr [-i] [-p] [-h]")
    print("-i: 输入源, 可以为以jpg结尾到图片，也可以是摄像头操作符(0, 1, ...)")
    print("-p: 图像处理参数, 格式为kernel_cols,kernel_rows,erode_iterations,dilate_iterations, 不要有空格，每个数字均为整数")
    print("-g: 采集间隔")
    print("-h: 显示帮助")


def show_help():
    print_usage()
    sys.exit()


def show_err():
    print("输入参数错误！")
    show_help()


def parse_args(args):
    if len(args) > 1:
        opts, args = getopt.getopt(sys.argv[1:], "hi:p:g:")
        l_input_file = None
        l_erode_param = None
        l_inter_gap = 180
        l_cap_index = -1
        try:
            for op, value in opts:
                if op == "-i":
                    if value.endswith('.jpg'):
                        l_input_file = value
                    else:
                        l_cap_index = int(value)
                elif op == "-p":
                    params = value.split(',')
                    if len(params) == 4:
                        l_erode_param = (int(params[0]), int(params[1]), int(params[2]), int(params[3]))
                    else:
                        show_err()
                elif op == "-g":
                    l_inter_gap = int(value)
                    if l_inter_gap <= 10:
                        print('采集间隔需大于10秒')
                        show_help()
                elif op == "-h":
                    show_help()
                else:
                    show_err()
        except ValueError:
            show_err()
    else:
        filename = '1.2-晚8点燃气表.jpg'  # 'calculator01.jpg'
        l_input_file = 'data/inputs/' + filename
        l_erode_param = (6, 3, 3, 2)
        l_inter_gap = 180
        l_cap_index = -1
    return l_input_file, l_cap_index, l_inter_gap, l_erode_param


if __name__ == '__main__':
    logger_main = logging.getLogger("dmr.main")
    input_file, cap_index, inter_gap, erode_param = parse_args(sys.argv)
    if input_file is not None:
        area = load_area_ini()
        model = load_cnn('data/train_data/cnn')
        origin_img = cv2.imread(input_file, cv2.IMREAD_COLOR)
        res = dmr_number(origin_img, area, model, erode_param)
        logger_main.info('dmr result: ' + str(res))
    elif cap_index >= 0:
        area = load_area_ini()
        model = load_cnn('data/train_data/cnn')
        cap = cv2.VideoCapture(cap_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        # cap.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_BGR)
        try:
            while True:
                start_time = time.time()
                # with PiCamera() as camera:
                #     camera.resolution = (1024, 768)
                #     camera.start_preview()
                #     # 摄像头预热2秒
                #     time.sleep(2)
                #     with picamera.array.PiRGBArray(camera) as stream:
                #         camera.capture(stream, format='bgr')
                #         # 此时就可以获取到bgr的数据流了
                #         origin_img = stream.array
                # cv2.waitKey(1000)
                ret, origin_img = cap.read()
                # origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite('data/train_data/test.jpg', origin_img)
                plt_img(0, origin_img)
                res = dmr_number(origin_img, area, model, erode_param)
                logger_main.info('dmr result: ' + str(res))
                end_time = time.time()
                used_time = end_time - start_time
                logger_main.info('time used: ' + str(used_time) + 's')
                if used_time < inter_gap:
                    time.sleep(inter_gap - used_time)
        except EnvironmentError:
            cap.release()
