# -*- codeing = utf-8 -*-
# @Time : 2021/08/22 20:24
# @Author : 217703 ZHANG WENXUAN
# @File : test.py
# @Software : PyCharm
import torch
import torchvision

print(torch.cuda.is_available())
print('torch_cuda: ', torch.version.cuda)
print(torch.__version__)
print(torchvision.__version__)

# conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge


import pynput
from pynput.keyboard import Listener
import pyautogui
# conda install -c conda-forge pyautogui

import time

t0 = time.time()
# for i in range(10):
#     pyautogui.moveRel(50, 0)
# print('pyautogui move', time.time() - t0)

mouse = pynput.mouse.Controller()
for i in range(10):
    mouse.move(50, 0)
print('pynput move', time.time() - t0)


def press(key):
    print(key)  # 输出键位

def start_listen():
    with Listener(on_press=press) as listener:
        listener.join()
    # 监听键盘按键

if __name__ == '__main__':
    start_listen()
