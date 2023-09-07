import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def nothing(x):  # 滑动条的回调函数
    pass



cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)  # 建立空窗口



cv2.createTrackbar('threshold', "img", 0, 60, nothing)  # 创建滑动条

def img_perspect_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size)

def cal_perspective_params(img, points):
    offset_x = int(img.shape[1]/4)
    offset_y = 0
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(points)
    # 设置俯视图中的对应的四个点
    dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x, offset_y],
                      [offset_x, img_size[1] - offset_y], [img_size[0] - offset_x, img_size[1] - offset_y]])
    # 原图像转换到俯视图
    M = cv2.getPerspectiveTransform(src, dst)
    # 俯视图到原图像
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    return M, M_inverse
def enhance_contrast(grayimg):
    # 读取图像

    gray_img = grayimg
    # 创建自适应直方图均衡化对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 应用自适应直方图均衡化
    enhanced_img = clahe.apply(gray_img)

    return enhanced_img

def tagmask(image):
    # 转换图像为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义蓝色范围
    lower_blue = np.array([125, 43, 46])  # 蓝色的下限
    upper_blue = np.array([150, 255, 255])  # 蓝色的上限

    # 创建蓝色掩膜
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 进行形态学操作，去除噪声和填充线条
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.erode(blue_mask, kernel, iterations=1)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)
    # cv2.imshow("bluemask",blue_mask)
    return  blue_mask

def pipeline(img, s_thresh=(170, 255), sx_thresh=(40, 200)):
    # 复制原图像
    img =img


    # 颜色空间转换
    # 将图像转换为HSL色彩空间
    hsl_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hsl_image = cv2.GaussianBlur(hsl_image, (3, 3), 0)

    # 分离H、S、L通道
    h_channel, l_channel, s_channel = cv2.split(hsl_image)
    cv2.imshow("l",l_channel)

    # 进行双边滤波
    d = 15  # 邻域直径
    sigma_color = 25  # 颜色空间标准差
    sigma_space = 75  # 坐标空间标准差
    l_channelf = cv2.bilateralFilter(l_channel, d, sigma_color, sigma_space)
    # sobel边缘检测
    fl_sobelx = cv2.Sobel(l_channelf, cv2.CV_64F, 1, 0, ksize=3)  # 在x方向上计算一阶导数
    fl_sobelx = cv2.convertScaleAbs(fl_sobelx)
    cv2.imshow("fl_soblex",fl_sobelx)

    # sobel边缘检测
    l_sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)  # 在x方向上计算一阶导数
    l_sobelx = cv2.convertScaleAbs(l_sobelx)
    # cv2.imshow("l_soblex",l_sobelx)

    # # 进行双边滤波
    # d = 15  # 邻域直径
    # sigma_color = 75  # 颜色空间标准差
    # sigma_space = 75  # 坐标空间标准差
    # l_sobelxf = cv2.bilateralFilter(l_sobelx, d, sigma_color, sigma_space)
    # cv2.imshow("l_soblexf", l_sobelxf)
    #
    #
    # sobel边缘检测
    # sobel边缘检测
    fl_sobely = cv2.Sobel(l_channelf, cv2.CV_64F, 0, 1, ksize=3)  # 在y方向上计算一阶导数
    fl_sobely = cv2.convertScaleAbs(fl_sobely)
    cv2.imshow("fl_sobley", fl_sobely)


    fl_sobelx_and_fl_sobely = cv2.bitwise_or(fl_sobelx, fl_sobely)

    cv2.imshow("fl_sobelx_and_fl_sobely", fl_sobelx_and_fl_sobely)












    # 对梯度幅值进行二值化
    threshold_value = 20  # 阈值
    _, thresholded_sobel_fl_sobelx_and_fl_sobely = cv2.threshold(fl_sobelx_and_fl_sobely, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresholded_sobel_fl_sobelx_and_fl_sobely", thresholded_sobel_fl_sobelx_and_fl_sobely)




    # 将二值图像取反
    bthresholded_sobel_fl_sobelx_and_fl_sobely = cv2.bitwise_not(thresholded_sobel_fl_sobelx_and_fl_sobely)
    cv2.imshow('bthresholded_sobel_fl_sobelx_and_fl_sobely', bthresholded_sobel_fl_sobelx_and_fl_sobely)

    # 创建一个黑色背景图像作为绘制轮廓的画布
    canvas = np.zeros_like(frame)


    # 连接连通区域并获取统计信息
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_sobel_fl_sobelx_and_fl_sobely)

    # 绘制连接后的连通区域
    for i in range(1, num_labels):  # 跳过背景（标签为0）
        x, y, w, h, area = stats[i]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示绘制了连通区域的图像
    cv2.imshow('Connected Components', canvas)
    cv2.imshow('Contours', canvas)
    # 查找图像中的轮廓
    contours, _ = cv2.findContours(thresholded_sobel_fl_sobelx_and_fl_sobely, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # 计算轮廓的最小外接矩形并绘制在原始图像上
    for contour in contours:
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 绘制旋转矩形
        cv2.drawContours(l_channelf, [box], 0, (255, 255, 255), 2)

        # 计算旋转矩形的宽度和高度
        width = np.linalg.norm(box[0] - box[1])  # 计算两点之间的欧氏距离作为宽度
        height = np.linalg.norm(box[1] - box[2])  # 计算两点之间的欧氏距离作为高度

        # 计算长宽比
        aspect_ratio = width / height

        # 计算面积
        area = width * height

        # if aspect_ratio > 8 and area > 100:
        #     # 绘制旋转矩形
        #     cv2.drawContours(subtracted_image1, [box], 0, (255, 255, 255), 2)

    cv2.imshow("l_channelf", l_channelf)



    threshold = 100 + 2 * cv2.getTrackbarPos('threshold', "img")  # 获取滑动条值
    # cloneframe = l_channelf.copy()
    # lines = cv2.HoughLines(thresholded_sobel_fl_sobelx_and_fl_sobely, 1, np.pi / 180, threshold)
    #
    # if lines is not None:
    #     for line in lines:
    #         rho = line[0][0]
    #         theta = line[0][1]
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         x1 = int(x0 + 1000 * (-b))
    #         y1 = int(y0 + 1000 * (a))
    #         x2 = int(x0 - 1000 * (-b))
    #         y2 = int(y0 - 1000 * (a))
    #
    #         cv2.line(cloneframe, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    # cv2.imshow('clone Image', cloneframe)

    # cloneframe = frame.copy()
    # # 使用霍夫直线检测找到图像中的直线
    # lines = cv2.HoughLinesP(thresholded_sobel_fl_sobelx_and_fl_sobely, rho=1, theta=np.pi / 180, threshold=100, minLineLength=10, maxLineGap=40)
    #
    # # 筛选直角边界框
    # right_angle_boxes = []
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         if abs(x2 - x1) > 0 and abs(y2 - y1) > 0:
    #             angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    #             if 85 <= abs(angle) <= 95:
    #                 right_angle_boxes.append(line[0])
    #
    #     # 在原始图像上绘制直角边界框
    #     for box in right_angle_boxes:
    #         x1, y1, x2, y2 = box
    #         cv2.rectangle(cloneframe, (x1, y1), (x2, y2), (255, 255, 255), 2)
    #
    # # 显示原始灰度图像和绘制了直角边界框的图像
    # cv2.imshow('clone Image', cloneframe)




    l_sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)  # 在y方向上计算一阶导数
    l_sobely = cv2.convertScaleAbs(l_sobely)
    # cv2.imshow("l_sobley", l_sobely)

    #
    #
    # l_sobelyf = cv2.bilateralFilter(l_sobely, d, sigma_color, sigma_space)
    # cv2.imshow("l_sobleyf", l_sobelyf)

    # # 将图像相减
    # subtracted_image1 = cv2.subtract(fl_sobelx, fl_sobely)
    # cv2.imshow("subx-y", subtracted_image1)
    #
    # subtracted_image2 = cv2.subtract(fl_sobely, fl_sobelx)
    # cv2.imshow("suby-x", subtracted_image2)










    # 对梯度幅值进行二值化
    threshold_value = 50  # 阈值
    _, thresholdedSobelx = cv2.threshold(l_sobelx, threshold_value, 255, cv2.THRESH_BINARY)
    # cv2.imshow("l_thresholdSobel", thresholdedSobelx)








    # 对梯度幅值进行二值化
    threshold_value = 100  # video阈值
    _, thresholdedL_channel = cv2.threshold(l_channel, threshold_value, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresholdedL_channel", thresholdedL_channel)

    # 合并多个二值图像的并集
    result = cv2.bitwise_and(thresholdedL_channel, cv2.bitwise_or(thresholdedSobelx, thresholdedSobelx))
    return result



# 精确定位车道线
def cal_line_param(binary_warped):
    # 1.确定左右车道线的位置
    # 统计直方图
    max = binary_warped.shape
    # print(max)
    # print(max)
    histogram = np.sum(binary_warped[:, :], axis=0)

    #
    # 在统计结果中找到左右最大的点的位置，作为左右车道检测的开始点
    # 将统计结果一分为二，划分为左右两个部分，分别定位峰值位置，即为两条车道的搜索位置
    midpoint = int(histogram.shape[0] /2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 2.滑动窗口检测车道线
    # 设置滑动窗口的数量，计算每一个窗口的高度
    nwindows = 9
    window_height = int(binary_warped.shape[0] / nwindows)
    # print(window_height)
    # 获取图像中不为0的点
    # 在这个代码示例中，binary_warped
    # 是一个二值化的图像。为了获取图像中非零（不为0）的点，你可以使用np.nonzero()
    # 函数。这个函数返回数组中非零元素的索引。
    # nonzero[0]返回行坐标，nonzero[1]返回列坐标
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # 车道检测的当前位置
    leftx_current = leftx_base
    rightx_current = rightx_base
    # 设置x的检测范围，滑动窗口的宽度的一半，手动指定
    margin = 100
    # 设置最小像素点，阈值用于统计滑动窗口区域内的非零像素个数，小于50的窗口不对x的中心值进行更新
    minpix = 50
    # 用来记录搜索窗口中非零点在nonzeroy和nonzerox中的索引
    left_lane_inds = []
    right_lane_inds = []

    # 遍历该副图像中的每一个窗口
    for window in range(nwindows):
        # 设置窗口的y的检测范围，因为图像是（行列）,shape[0]表示y方向的结果，上面是0
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        # 左车道x的范围
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        # 右车道x的范围
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin


        # 确定非零点的位置x,y是否在搜索窗口中，将在搜索窗口内的x,y的索引存入left_lane_inds和right_lane_inds中
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]


        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 如果获取的点的个数大于最小个数，则利用其更新滑动窗口在x轴的位置
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # 将检测出的左右车道点转换为array
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 获取检测出的左右车道点在图像中的位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    color = (255, 255, 255)
    # 在图像上绘制左车道点
    for x, y in zip(leftx, lefty):
        cv2.circle(binary_warped, (x, y), 3,color , -1)

    # 在图像上绘制右车道点
    for x, y in zip(rightx, righty):
        cv2.circle(binary_warped, (x, y), 3, color, -1)

    if len(lefty) == 0 or len(leftx) == 0:
        left_fit = [-1,-1,-1]

    else:
        # 3.用曲线拟合检测出的点,二次多项式拟合，返回的结果是系数
        left_fit = np.polyfit(lefty, leftx, 2)

    if len(righty) == 0 or len(rightx) == 0:

        right_fit = [-1,-1,-1]
    else:
    # 3.用曲线拟合检测出的点,二次多项式拟合，返回的结果是系数
        right_fit = np.polyfit(righty, rightx, 2)

    print(left_fit)
    print(right_fit)

    return left_fit , right_fit
def draw(transform_img,left_fit,right_fit):
    if all(elem == -1 for elem in left_fit) or all(elem == -1 for elem in right_fit):
        print("列表全是 -1，退出函数")
        return transform_img
    # 创建一个空白图像
    image = transform_img

    # 设置线的颜色（BGR格式）
    color = (0, 0, 255)

    # 设置线的宽度
    thickness = 2

    # 计算拟合曲线上的点
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    # 将拟合曲线上的点转换为整数坐标
    left_points = np.array([np.column_stack((left_fitx, ploty))], dtype=np.int32)
    right_points = np.array([np.column_stack((right_fitx, ploty))], dtype=np.int32)

    # 在图像上绘制左车道线和右车道线
    cv2.polylines(image, left_points, isClosed=False, color=color, thickness=thickness)
    cv2.polylines(image, right_points, isClosed=False, color=color, thickness=thickness)

    return image




if __name__ == "__main__":
    # cap = cv2.VideoCapture('project_video.mp4')
    # cap = cv2.VideoCapture("./videodata/out728P.mp4" )
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error opening video file")
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break


        # 在这里对帧进行处理
        #
        # cv2.imshow('Frame', frame)
        # img = frame.copy()
        # # 将图像调整为360x640大小
        #
        # # a = cv2.resize(img, (640, 360))
        # # 缩小图像的尺寸
        # # a = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        #
        # a = img[480-360:480, 0:640]
        # cv2.imshow('Framea', a)
        #
        #
        #
        #
        # points = [[601, 448], [683, 448], [230, 717], [1097, 717]]
        # pointsa = [[601//2, 448//2], [683//2, 448//2], [230//2, 717//2], [1097//2, 717//2]]
        #
        # M, M_inverse = cal_perspective_params(a, pointsa)
        # transform_img = img_perspect_transform(a, M)
        #
        #
        # cv2.imshow("transform",transform_img)

        # 提取ROI
        # 获取图像的高度和宽度
        height, width = frame.shape[:2]
        # print(frame.shape[:2])
        # 定义ROI的高度比例，这里设置为下部的一半
        roi_height_ratio = 0.2

        # 计算ROI的高度
        roi_height = int(height * roi_height_ratio)
        bias =60
        # 提取ROI
        roi1 = frame[height - roi_height-bias:height-bias, 0:width]

        # # 矩形的左上角和右下角坐标
        # pt1 = (0, height - roi_height-bias)
        # pt2 = (width, height-bias)
        #
        # # 绘制蓝色矩形
        # color = (255, 0, 0)  # 蓝色 (BGR)
        # thickness = 2
        # cv2.rectangle(frame, pt1, pt2, color, thickness)

        roi2 = frame[height - roi_height*2 - bias:height - bias-roi_height, 0:width]
        roi3 = frame[height - roi_height*3 - bias:height - bias-roi_height*2, 0:width]
        # roi = frame[frame.shape[0]//2:frame.shape[0], 0::frame.shape[1]]
        cv2.imshow("frame", frame)
        #
        # cv2.imshow("roi1", roi1)
        # cv2.imshow("roi2", roi2)
        # cv2.imshow("roi3", roi3)
        # cv2.imshow("mask", tagmask(frame))
        cv2.imshow("pipeline",pipeline(frame))
        # cv2.imshow("pipeline",pipeline(roi1))
        # cv2.imshow("pipeline",pipeline(roi2))
        # cv2.imshow("pipeline",pipeline(roi3))

        # # 拟合车道线
        # left_fit, right_fit = cal_line_param(pipeline(transform_img))
        #
        # image = draw(transform_img,left_fit,right_fit)
        # # 显示绘制了车道线的图像
        # cv2.imshow('Lane Lines', image)
        # 按下 'q' 键退出循环
        # cv2.waitKey(100)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()







4