# -*- codeing = utf-8 -*-
# @Time : 2021/08/21 22:35
# @Author : 217703 ZHANG WENXUAN
# @File : cs_model.py
# @Software : PyCharm

import torch
from models.experimental import attempt_load

# detect.py

device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'

weights = r'C:\Users\Zman\PycharmProjects\yolov5\aim-csgo\models\yolov5s.pt'
imgsz = 640


def load_model():
    # p76
    model = attempt_load(weights, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16
    if device != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    return model
