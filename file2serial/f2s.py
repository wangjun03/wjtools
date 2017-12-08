#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Wang Jun
# @Contact: wangjun@tsingcon.com
# @Date  : 2017/12/6
# @license : Copyright(C), Beijing Tsing Con Technology Co., Ltd.
# @Desc  : 
# -*- coding:utf-8 -*-
import sys
import os
import configparser as parser
from PyQt5 import QtCore, QtGui, QtWidgets
from os.path import isfile
import datetime
import serial

from f2s_ui import Ui_MainWindow

CONFIG_FILE_PATH = "f2s.cfg"


def send_cmd(cmd, com_no='COM3', baud_rate=115200):
    result = []
    res_state = False
    try:
        with serial.Serial(com_no, baud_rate, timeout=1) as ser:
            ser.write(str.encode(cmd))
            response = str(ser.readline(), encoding='gbk')
            while response:
                result.append(response)
                response = str(ser.readline(), encoding='gbk')
            ser.close()
            res_state = True
    except serial.serialutil.SerialException as se:
        result.append("打开串口失败！")
        result.append(str(se))
    return result, res_state


def check_config_file():
    if not os.path.exists(CONFIG_FILE_PATH):
        f = open(CONFIG_FILE_PATH, mode="w", encoding="UTF-8")
        f.close()


def get_config(config, selection, option, default=""):
    if config is None:
        return default
    else:
        try:
            return config.get(selection, option)
        except:
            return default


def write_config(config, selection, option, value):
    if not config.has_section(selection):
        config.add_section(selection)
    config.set(selection, option, value)


class WaqsApp(QtWidgets.QMainWindow):
    def __init__(self):
        check_config_file()
        self.config = parser.ConfigParser()
        self.config.read(CONFIG_FILE_PATH)
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("未打开配置")
        self.read_settings()
        self.ui.statusbar.showMessage("未打开配置")
        self.com_no = get_config(self.config, "COMM", "default_com", "COM3")
        self.baud_rate = int(get_config(self.config, "COMM", "default_baudrate", "115200"))
        self.coms = get_config(self.config, "COMM", "allow_com", "COM3")
        self.bauds = get_config(self.config, "COMM", "allow_baudrate", "115200")
        for com_no in self.coms.split(','):
            self.ui.comboBox_com.addItem(com_no)
        self.ui.comboBox_com.setCurrentText(self.com_no)
        for baud in self.bauds.split(','):
            self.ui.comboBox_baud.addItem(baud)
        self.ui.comboBox_baud.setCurrentText(str(self.baud_rate))
        # com_group = QtWidgets.QActionGroup(self)
        # self.ui.actionCOM1.setActionGroup(com_group)
        # self.ui.actionCOM2.setActionGroup(com_group)
        # self.ui.actionCOM3.setActionGroup(com_group)
        # self.ui.actionCOM4.setActionGroup(com_group)
        # self.ui.actionCOM5.setActionGroup(com_group)
        # self.ui.actionCOM6.setActionGroup(com_group)
        # self.ui.actionCOM7.setActionGroup(com_group)
        # self.ui.actionCOM1
        # botte_group = QtWidgets.QActionGroup(self)
        # self.ui.action9600.setActionGroup(botte_group)
        # self.ui.action115200.setActionGroup(botte_group)
        self.ui.actiondakai.triggered.connect(self.file_dialog)
        self.ui.actionupload.triggered.connect(self.upload_cfg)
        self.filename = ""

    def read_settings(self):
        # 宽度 高度
        width = get_config(self.config, "Display", "width", "1000")
        height = get_config(self.config, "Display", "height", "600")
        size = QtCore.QSize(int(width), int(height))

        # 屏幕位置
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        pos_x = get_config(self.config, "Display", "x", (screen.width() - 1000) // 2)
        pos_y = get_config(self.config, "Display", "y", (screen.height() - 600) // 2)
        pos = QtCore.QPoint(int(pos_x), int(pos_y))

        # 是否自动换行
        wrapMode = get_config(self.config, "TextEdit", "wrapmode", "True")

        # 字体
        fontFamile = get_config(self.config, "TextEdit", "font", "Consolas")
        fontSize = get_config(self.config, "TextEdit", "size", '14')
        fonts = QtGui.QFont(fontFamile, int(fontSize))

        self.resize(size)
        self.move(pos)
        self.ui.textBrowser.setLineWrapMode(wrapMode == "True")
        self.ui.textBrowser.setFont(fonts)

    def write_settings(self):
        self.com_no = self.ui.comboBox_com.currentText()
        self.baud_rate = int(self.ui.comboBox_baud.currentText())
        # 宽度、高度
        write_config(self.config, "Display", "height", str(self.size().height()))
        write_config(self.config, "Display", "width", str(self.size().width()))
        # 位置
        write_config(self.config, "Display", "x", str(self.pos().x()))
        write_config(self.config, "Display", "y", str(self.pos().y()))
        # 自动换行
        write_config(self.config, "TextEdit", "wrapmode",
                     str(self.ui.textBrowser.lineWrapMode() == QtWidgets.QTextBrowser.WidgetWidth))
        # 字体
        write_config(self.config, "TextEdit", "font", self.ui.textBrowser.font().family())
        # 大小
        write_config(self.config, "TextEdit", "size", str(self.ui.textBrowser.font().pointSize()))

        write_config(self.config, "COMM", "default_com", str(self.com_no))

        write_config(self.config, "COMM", "default_baudrate", str(self.baud_rate))

        write_config(self.config, "COMM", "allow_com", str(self.coms))

        write_config(self.config, "COMM", "allow_baudrate", str(self.bauds))

        # 回写
        self.config.write(open(CONFIG_FILE_PATH, "w"))

    def file_dialog(self):
        self.filename, filetype = QtWidgets.QFileDialog.getOpenFileName(self, "选择配置文件",
                                                                        filter="Text Files (*.txt)")
        if isfile(self.filename):
            s = open(self.filename, 'r').read()
            self.ui.textBrowser.setText(s)
            self.setWindowTitle(self.filename)
            self.ui.statusbar.showMessage("成功打开配置文件:" + self.filename)

    def upload_cfg(self):
        self.com_no = self.ui.comboBox_com.currentText()
        self.baud_rate = int(self.ui.comboBox_baud.currentText())
        if self.filename != "":
            s = ''
            fd = open(self.filename, 'r')
            cmd = fd.readline()
            res_state = True
            while cmd and res_state:
                if not cmd.startswith('#'):
                    s += datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' cmd: ' + cmd + '\n'
                    results, res_state = send_cmd(cmd, self.com_no, self.baud_rate)
                    for result in results:
                        s += datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' rep: ' + result + '\n'
                cmd = fd.readline()
            self.ui.textBrowser.setText(s)
            if res_state:
                self.ui.statusbar.showMessage("成功上传配置:" + self.filename)
            else:
                self.ui.statusbar.showMessage("上传配置失败:" + self.filename)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    waqs_app = WaqsApp()
    waqs_app.show()
    code = app.exec_()
    waqs_app.write_settings()
    sys.exit(code)
