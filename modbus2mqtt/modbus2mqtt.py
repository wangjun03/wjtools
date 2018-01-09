#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Wang Jun
# @Contact: wangjun@tsingcon.com
# @Date  : 2017/12/28
# @Desc  :
import serial
import json
import modbus_tk.utils
import modbus_tk.modbus
import modbus_tk.defines as cst
import modbus_tk.exceptions
from modbus_tk import modbus_rtu
import time
import paho.mqtt.client as mqtt

# PORT = 1
# PORT = "COM5"
# "/dev/ttyUSB0"
logger = modbus_tk.utils.create_logger("console")
mqtt_north = mqtt.Client()



def main():
    """main"""


    try:
        # Connect to the slave
        master = modbus_rtu.RtuMaster(
            serial.Serial(port="COM5", baudrate=9600, bytesize=8, parity='N', stopbits=1, xonxoff=0)
        )
        master.set_timeout(2.0)
        master.set_verbose(True)
        logger.info("connected")
        logger.info(master.execute(3, cst.READ_HOLDING_REGISTERS, 19, 2))
        # for i in range(1, 255):
        #     try:
        #         logger.info(master.execute(i, cst.READ_HOLDING_REGISTERS, 19, 2))
        #     except modbus_tk.exceptions.ModbusInvalidResponseError:
        #         pass
        # send some queries
        # logger.info(master.execute(1, cst.READ_COILS, 0, 10))
        # logger.info(master.execute(1, cst.READ_DISCRETE_INPUTS, 0, 8))
        # logger.info(master.execute(1, cst.READ_INPUT_REGISTERS, 100, 3))
        # logger.info(master.execute(1, cst.READ_HOLDING_REGISTERS, 100, 12))
        # logger.info(master.execute(1, cst.WRITE_SINGLE_COIL, 7, output_value=1))
        # logger.info(master.execute(1, cst.WRITE_SINGLE_REGISTER, 100, output_value=54))
        # logger.info(master.execute(1, cst.WRITE_MULTIPLE_COILS, 0, output_value=[1, 1, 0, 1, 1, 0, 1, 1]))
        # logger.info(master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 100, output_value=xrange(12)))

    except modbus_tk.modbus.ModbusError as exc:
        logger.error("%s- Code=%d", exc, exc.get_exception_code())


def load_cfg(filename="m2m.cfg.txt"):
    fd = open(filename, 'r')
    cfg_str = fd.read()
    cfg = json.loads(cfg_str)
    return cfg


def read_one_port(port_cfg):
    try:
        # Connect to the slave
        master = modbus_rtu.RtuMaster(
            serial.Serial(port=port_cfg['port'],
                          baudrate=port_cfg['baudrate'],
                          bytesize=port_cfg['bytesize'],
                          parity=port_cfg['parity'],
                          stopbits=port_cfg['stopbits'],
                          xonxoff=port_cfg['xonxoff'])
        )
        master.set_timeout(port_cfg['timeout'])
        master.set_verbose(True)
        logger.info("connected")
        data_list = []
        for data_cfg in port_cfg['data']:
            try:
                result = master.execute(data_cfg['slave_id'], cst.READ_HOLDING_REGISTERS, data_cfg['addr'], data_cfg['len'])
                # result = (1, 2)
                data_list.extend(list(result))
                time.sleep(1)
            except modbus_tk.exceptions.ModbusInvalidResponseError:
                for i in range(data_cfg['len']):
                    data_list.append(0)
    except modbus_tk.modbus.ModbusError as exc:
        data_list = None
        logger.error("%s- Code=%d", exc, exc.get_exception_code())
    except serial.serialutil.SerialException as se:
        data_list = None
        logger.error(se)
    return data_list


def connect_mqtt(mqttc, cfg):
    mqttc.connect(cfg['mqtt_host'], port=cfg['mqtt_port'], keepalive=cfg['keepalive'], bind_address=cfg['bindaddress'])


# def on_publish(client, userdata, mid):
#     logger.info(time.time(), "published a msg")


def main_loop(cfg):
    port_num = len(cfg['data'])
    seq = 0
    freq = cfg['freq']

    if cfg['username'] != "":
        mqtt_north.username_pw_set(cfg['username'], password=cfg['password'])
    connect_mqtt(mqtt_north, cfg)
    for port_cfg in cfg['data']:
        seq += 1
        data_list = read_one_port(port_cfg)
        if data_list is not None:
            msg = {
                "pro": cfg['proj'],
                "id": cfg['id'],
                "ts": int(time.time()),
                "freq": freq,
                "num": port_num,
                "seq": seq,
                "valid": "1",
                "volt": "15",
                "Data": data_list
            }
            msg_str = json.dumps(msg)
            mqtt_north.publish(cfg['topic'], msg_str)

    close_mqtt(mqtt_north)


def close_mqtt(mqttc):
    mqttc.disconnect()


if __name__ == "__main__":
    # main()
    cfg = load_cfg()
    freq = cfg['freq']
    # mqtt_north.on_publish = on_publish
    while True:
        start_time = time.time()
        main_loop(cfg)
        end_time = time.time()
        used_time = end_time-start_time
        logger.info("used time: "+str(used_time)+"s")
        if used_time < freq * 60:
            time.sleep(freq * 60 - used_time)
