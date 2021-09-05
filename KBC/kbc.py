# -*- codeing = utf-8 -*-
# @Time : 2021/08/31 22:38
# @Author : 217703 ZHANG WENXUAN
# @File : kbc.py
# @Software : PyCharm
import time

import pynput
from pynput.keyboard import Key,Controller,Listener

ctr = pynput.keyboard.Controller()
ctr2 = Controller()

def press(key):
    print(key)  # 输出键位
    K = str(key)
    if(K == 'Key.ctrl_l'):
        # ctr.type('w')
        # ctr.type([pynput.keyboard.Key.esc])
        with ctr2.pressed("w"):  # 按下a
            ctr2.press("d")
            time.sleep(3)
            ctr2.release("d")  # 松开a
        return False

    if(K == 'Key.ctrl_r'):
        return False

def start_listen():
    with Listener(on_press=press) as listener:
        listener.join()

    # 监听键盘按键

if __name__ == '__main__':
    start_listen()
