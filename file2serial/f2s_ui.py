# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'f2s_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 768)
        MainWindow.setMaximumSize(QtCore.QSize(1280, 768))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMaximumSize(QtCore.QSize(1280, 725))
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(0, 25, 1280, 700))
        self.textBrowser.setMaximumSize(QtCore.QSize(1280, 725))
        self.textBrowser.setObjectName("textBrowser")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(-1, 0, 403, 26))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.comboBox_com = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.comboBox_com.setObjectName("comboBox_com")
        self.horizontalLayout.addWidget(self.comboBox_com)
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.comboBox_baud = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.comboBox_baud.setObjectName("comboBox_baud")
        self.horizontalLayout.addWidget(self.comboBox_baud)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actiondakai = QtWidgets.QAction(MainWindow)
        self.actiondakai.setIconVisibleInMenu(False)
        self.actiondakai.setObjectName("actiondakai")
        self.actionupload = QtWidgets.QAction(MainWindow)
        self.actionupload.setObjectName("actionupload")
        self.menu.addAction(self.actiondakai)
        self.menu_2.addAction(self.actionupload)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "无线硬件配置"))
        self.label.setText(_translate("MainWindow", "串口参数："))
        self.label_2.setText(_translate("MainWindow", "串口号："))
        self.label_3.setText(_translate("MainWindow", "波特率："))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "运行"))
        self.actiondakai.setText(_translate("MainWindow", "打开"))
        self.actionupload.setText(_translate("MainWindow", "上传配置"))

