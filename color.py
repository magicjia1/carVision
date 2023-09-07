# HSV调试
import cv2 as cv
import numpy as np


def nothing(x):
    pass


cap = cv.VideoCapture(2)
img = np.zeros((300, 512, 3), np.uint8)
cv.namedWindow('image')

cv.createTrackbar('H_L', 'image', 0, 360, nothing)
cv.createTrackbar('H_H', 'image', 0, 360, nothing)
cv.createTrackbar('S_L', 'image', 0, 255, nothing)
cv.createTrackbar('S_H', 'image', 0, 255, nothing)
cv.createTrackbar('V_L', 'image', 0, 255, nothing)
cv.createTrackbar('V_H', 'image', 0, 255, nothing)

while (1):
    # 读取帧
    _, frame = cap.read()
    H_L = cv.getTrackbarPos('H_L', 'image')
    H_H = cv.getTrackbarPos('H_H', 'image')
    S_L = cv.getTrackbarPos('S_L', 'image')
    S_H = cv.getTrackbarPos('S_H', 'image')
    V_L = cv.getTrackbarPos('V_L', 'image')
    V_H = cv.getTrackbarPos('V_H', 'image')

    # 转换颜色空间 BGR 到 HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # 定义HSV中颜色的范围

    lower_ = np.array([H_L, S_L, V_L])
    upper_ = np.array([H_H, S_H, V_H])
    # 设置HSV的阈值使得只取该范围内颜色
    mask = cv.inRange(hsv, lower_, upper_)
    # 将掩膜和图像逐像素相加
    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('image', res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
