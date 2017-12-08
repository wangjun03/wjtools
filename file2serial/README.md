# file2serial

file2serial用于将配置文件的文本逐条通过串口发送给接收设备，并记录命令发送和回复记录

file2serial串口使用pySerial模块，界面使用pyqt5

## 程序配置文件f2s.cfg

通信参数

    [COMM]
    default_com = COM3
    default_baudrate = 115200
    allow_com = COM1,COM3,COM4,COM5
    allow_baudrate = 9600,115200

default_com为默认串口号

default_baudrate为默认波特率

allow_com为可选串口号，可根据需要修改，对于linux或mac，可改为tty.xxxxx

allow_baudrate为可选波特率，可根据需要修改
