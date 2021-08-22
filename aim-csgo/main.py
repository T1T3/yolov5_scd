# -*- codeing = utf-8 -*-
# @Time : 2021/08/21 21:12
# @Author : 217703 ZHANG WENXUAN
# @File : main.py
# @Software : PyCharm
import win32con
import win32gui
import numpy as np
import cv2
import torch
from cs_model import load_model
from grabscreen import grab_screen
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox

# detect.py   VVVV

device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'

imgsz = 640
conf_thres = 0.4  # 物体のクラスである、conﬁdence loss
iou_thres = 0.05  # 画像の重なりの割合を表す値であり
x, y = (2560, 1440)  # プログラムウィンドウのサイズ
re_x, re_y = (2560, 1440)  # スクリーンのサイズ

# L76
model = load_model()
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names  # get class names

while True:
    img0 = grab_screen(region=(0, 0, x, y))  # 左上の角から右下の角
    img0 = cv2.resize(img0, (re_x, re_y))

    # Padded resize FROM datasets.py L220
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert    FROM datasets.py L223
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim  ==[img=img.unsqueeze(0)]

    pred = model(img, augment=False, visualize=False)[0]  # 予測

    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
    # (prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300)

    # print(pred)

    aims = []
    # detect.py-->L172
    for i, det in enumerate(pred):  # detections per image
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            # BBOX   VVVVVV　　バウンディングボックス
            for *xyxy, conf, cls in reversed(det):
                # if save_txt:   Write to file
                # save bbox:(tag,x_center,y_center,x_width,y_width)　目標検出ボックス
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format (only need else-->)

                aim = ('%g ' * len(line)).rstrip() % line
                # print(aim)  !type-->str!

                # make aim type[list]  len-->5
                # like [‘62’， ‘0.777148’，‘0.529861’， ‘0.437891’， ‘0.769444’]
                # [（種類番号）, アクシスx軸,  アクシスy軸,  ボックス長さ,   ボックス高さ]
                # 数字 -->　(スクリーンサイズの百分比)
                aim = aim.split(' ')
                # print(aim)
                aims.append(aim)

        if len(aims):
            for i, det in enumerate(aims):
                _, x_center, y_center, width, height = det  # 種類番号いらない X

                # 　バウンディングボックスを縮小する
                #  元のパラメータがstr型  -->  float
                x_center = re_x * float(x_center)
                y_center = re_y * float(y_center)
                width = re_x * float(width)
                height = re_y * float(height)

                # 必ずint型
                top_left = (int(x_center - width / 2.0), int(y_center - height / 2.0))
                bottom_right = (int(x_center + width / 2.0), int(y_center + height / 2.0))

                color = (0, 255, 0) # RGB
                cv2.rectangle(img0, top_left, bottom_right, color, thickness=3)  # パラメータ必ずint型  線のサイズ=3   1/3

    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('detect', re_x // 3, re_y // 3)  # ウィンドウのサイズ 1/3
    cv2.imshow('detect', img0)

    hwnd = win32gui.FindWindow(None, 'detect')  # ウィンドウを探す
    CVRECT = cv2.getWindowImageRect('detect')
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    # ウィンドウを常に最前面にする  、０が左上隅（すみ）に固まる 、[win32con.SWP_NOMOVE | win32con.SWP_NOSIZE]-->移動可能

    if cv2.waitKey(1) & 0xFF == ord('p'):  # キーボードの「p」ボタン押しと、ウィンドウをしまう///ショートカットキー
        cv2.destroyAllWindows()
        break
