# -*- codeing = utf-8 -*-
# @Time : 2021/08/21 15:23
# @Author : 217703 ZHANG WENXUAN
# @File : video2img.py
# @Software : PyCharm

import cv2


def save_image(addr, image, num):
    address = addr + 'img_' + str(num) + '.jpg'  # \ ---> /
    print(address)
    cv2.imwrite(address, image)


video_path = r'C:\Users\Zman\PycharmProjects\yolov5\labeling\video\train3.flv'
out_path = r'C:/Users/Zman/PycharmProjects/yolov5/labeling/images/'  # \ ---> /

is_all_frame = True  # すべてフレームを取ります
sta_frame = 1  # 始めいてフレーム
end_frame = 1000  # 最後のフレーム

time_interval = 8  # 時間の間隔

videocapture = cv2.VideoCapture(video_path)
# pip install opencv-contrib-python

success, frame = videocapture.read()

# print(success,frame)


i = 0
j = 0
while success:
    i += 1
    if i % time_interval == 0:
        if not is_all_frame:
            if sta_frame <= i <= end_frame:
                j += 1
                print('save frame', j)
                save_image(out_path, frame, j)
            elif i > end_frame:
                break
        else:
            j += 1
            print('save frame', j)
            save_image(out_path, frame, j)

    success, frame = videocapture.read()
