# HSV调试
import cv2
import numpy as np




def nothing(x):
    pass


cap = cv2.VideoCapture(2)
# cap = cv2.VideoCapture("./videodata/out7285_blue3.mp4")

img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image')

cv2.createTrackbar('H_L', 'image', 0, 360, nothing)
cv2.createTrackbar('H_H', 'image', 0, 360, nothing)
cv2.createTrackbar('S_L', 'image', 0, 255, nothing)
cv2.createTrackbar('S_H', 'image', 0, 255, nothing)
cv2.createTrackbar('V_L', 'image', 0, 255, nothing)
cv2.createTrackbar('V_H', 'image', 0, 255, nothing)

cv2.createTrackbar('lower_threshold', 'image', 0, 255, nothing)
cv2.createTrackbar('upper_threshold', 'image', 0, 255, nothing)

while (1):
    # 读取帧
    _, frame = cap.read()



    height, width = frame.shape[:2]

    # 定义ROI的高度比例，这里设置为下部的一半
    roi_height_ratio = 0.5

    # 计算ROI的高度
    roi_height = int(height * roi_height_ratio)
    bias = 60
    # 提取ROI
    roi1 = frame[height - roi_height - bias:height - bias, 0+int(width * 0.1):width-int(width * 0.1)]
    # cv2.imshow("roi1", roi1)

    # 矩形的左上角和右下角坐标
    pt1 = (0+int(width * 0.1), height - roi_height-bias)
    pt2 = (width-int(width * 0.1), height-bias)

    # 绘制蓝色矩形
    color = (255, 255, 255)
    thickness = 2
    # cv2.rectangle(frame, pt1, pt2, color, thickness)
    # cv2.imshow('Original Image', frame)



    # 分离BGR通道
    b, g, r = cv2.split(frame)
    # 显示原始图像和分离后的通道
    cv2.imshow('Blue Channel', b)
    # cv2.imshow('Green Channel', g)
    cv2.imshow('Red Channel', r)
    # 对灰度图像进行反色处理
    inverted_imager = 255 - r
    inverted_imageb = 255 - b
    cv2.imshow('Inverted Imager', inverted_imager)
    cv2.imshow('Inverted Imageb', inverted_imageb)




    # 将图像相减,得到蓝色区域变亮 其他区域变暗
    subtracted_image1 = cv2.subtract(inverted_imager, inverted_imageb)
    cv2.imshow("subx-y", subtracted_image1)

    # 使用 Canny 边缘检测算法
    edges = cv2.Canny(subtracted_image1, 5, 50)
    cv2.imshow('Edges', edges)

    lower_threshold = cv2.getTrackbarPos('lower_threshold', 'image')
    upper_threshold = cv2.getTrackbarPos('upper_threshold', 'image')
    t, rst = cv2.threshold(r, lower_threshold, upper_threshold, cv2.THRESH_BINARY)

    t, rst = cv2.threshold(r, 67, 255, cv2.THRESH_BINARY)
    cv2.imshow("rst", rst)



    # 取反原始mask图像
    inverted_mask = cv2.bitwise_not(rst)
    cv2.imshow("rstr", inverted_mask)

    # 执行膨胀操作
    erode_imagerst = cv2.erode(inverted_mask, np.ones((7, 7), np.uint8), iterations=1)
    cv2.imshow("erodeterst", erode_imagerst)

    # 将取反后的mask图像与原始图像相与，得到新的mask图像
    subtracted_image2 = cv2.bitwise_and(subtracted_image1, subtracted_image1, mask=erode_imagerst)

    # #
    # # subtracted_image2 = cv2.subtract(subtracted_image1, rst)
    cv2.imshow("subtracted_image2", subtracted_image2)


    # lower_threshold = cv2.getTrackbarPos('lower_threshold', 'image')
    # upper_threshold = cv2.getTrackbarPos('upper_threshold', 'image')
    # # 设置亮度范围
    # lower_threshold = 100
    # upper_threshold = 200


    #
    # # 进行双边滤波
    # d = 25  # 邻域直径
    # sigma_color = 75  # 颜色空间标准差
    # sigma_space = 25  # 坐标空间标准差
    # fsubtracted_image1 = cv2.bilateralFilter(subtracted_image1, d, sigma_color, sigma_space)
    # cv2.imshow("fsubx-y", fsubtracted_image1)



    # # 自适应阈值二值化，只有一个返回值
    # dst = cv2.adaptiveThreshold(fsubtracted_image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 0)
    # cv2.imshow("dst", dst)


    # # 创建一个腐蚀核
    # kernel = np.ones((3,3), np.uint8)
    #
    # # 使用腐蚀操作消除小白点
    # eroded_img = cv2.erode(dst, kernel, iterations=1)
    # cv2.imshow("eroded_img", eroded_img)
    # # 膨胀
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # ldilate = cv2.dilate(dst, kernel, iterations=1)
    # cv2.imshow("ldilate", ldilate)
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # lopen = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel, iterations=2)
    # cv2.imshow("lopen", lopen)
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # lclose = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, iterations=2)
    # cv2.imshow("lclose", lclose)












    H_L = cv2.getTrackbarPos('H_L', 'image')
    H_H = cv2.getTrackbarPos('H_H', 'image')
    S_L = cv2.getTrackbarPos('S_L', 'image')
    S_H = cv2.getTrackbarPos('S_H', 'image')
    V_L = cv2.getTrackbarPos('V_L', 'image')
    V_H = cv2.getTrackbarPos('V_H', 'image')

    # 转换颜色空间 BGR 到 HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 定义HSV中颜色的范围

    lower_ = np.array([H_L, S_L, V_L])
    upper_ = np.array([H_H, S_H, V_H])
    blue_lower_ = np.array([63, 132, 0])
    blue_upper_ = np.array([130, 255, 199])

    # 设置HSV的阈值使得只取该范围内颜色
    mask = cv2.inRange(hsv, lower_, upper_)
    mask = cv2.inRange(hsv, blue_lower_, blue_upper_)
    # 将掩膜和图像逐像素相加
    res = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow('frame', frame)
    # cv2.imshow('mask', mas图像自身多次相加k)
    cv2.imshow('image', res)

    # 将彩色图像转换为灰度图像
    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)


    cv2.imshow("gray", gray_image)

    # 定义腐蚀和膨胀的结构元素（内核）
    kernel = np.ones((3, 3), np.uint8)

    eroded_image = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)



    # 执行腐蚀操作
    eroded_image = cv2.erode(gray_image, eroded_image, iterations=1)

    cv2.imshow("eroded_image", eroded_image)

    # 执行膨胀操作
    dilated_image = cv2.dilate(eroded_image, np.ones((5, 5), np.uint8), iterations=1)
    cv2.imshow("dilate", dilated_image)



    # 灰度图像相加
    result_image = cv2.add(1*subtracted_image1, 1*dilated_image)

    cv2.imshow("result_image", result_image)























    # 对图像进行二值化处理
    ret, binary_image = cv2.threshold(result_image, 23, 255, cv2.THRESH_BINARY)
    cv2.imshow("b", binary_image)



    # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3 ,3))

    ldilate = cv2.dilate(binary_image, kernel, iterations=1)

    cv2.imshow("bl", ldilate)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(ldilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算轮廓的最小外接矩形并绘制在原始图像上
    for contour in contours:
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # # 绘制旋转矩形
        # cv2.drawContours(subtracted_image1, [box], 0, (255, 255, 255), 2)

        # 计算旋转矩形的宽度和高度
        width = np.linalg.norm(box[0] - box[1])  # 计算两点之间的欧氏距离作为宽度
        height = np.linalg.norm(box[1] - box[2])  # 计算两点之间的欧氏距离作为高度

        # 计算长宽比
        aspect_ratio = width / height

        # 计算面积
        area = width * height

        if aspect_ratio >8 and area >100:
            # 绘制旋转矩形
            cv2.drawContours(subtracted_image1, [box], 0, (255, 255, 255), 2)

    cv2.imshow("subx-y", subtracted_image1)



    cv2.waitKey(10)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
