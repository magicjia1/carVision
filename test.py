import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def img_perspect_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size)

def cal_perspective_params(img, points):
    offset_x = 330
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

def pipeline(img, s_thresh=(170, 255), sx_thresh=(40, 200)):
    # 复制原图像
    img =img
    # 颜色空间转换
    # 将图像转换为HSL色彩空间
    hsl_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hsl_image = cv2.GaussianBlur(hsl_image, (3, 3), 0)

    # 分离H、S、L通道
    h_channel, l_channel, s_channel = cv2.split(hsl_image)
    # cv2.imshow("h",h_channel)
    # cv2.imshow("s",s_channel)
    # cv2.imshow("l",l_channel)


    # sobel边缘检测
    l_sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)  # 在x方向上计算一阶导数
    l_sobelx = cv2.convertScaleAbs(l_sobelx)
    # cv2.imshow("l_soblex",l_sobelx)


    # 对梯度幅值进行二值化
    threshold_value = 50  # 阈值
    _, thresholdedSobelx = cv2.threshold(l_sobelx, threshold_value, 255, cv2.THRESH_BINARY)
    # cv2.imshow("l_thresholdSobel", thresholdedSobelx)


    # s通道阈值处理
    s_sobel= cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=3)  # 在x方向上计算一阶导数
    s_sobelx = cv2.convertScaleAbs(s_sobel)
    # cv2.imshow("s_soblex", s_sobelx)



    # 对梯度幅值进行二值化
    threshold_value = 50  # 阈值
    _, thresholdedS_channel = cv2.threshold(s_sobelx, threshold_value, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresholdedS_channel", thresholdedS_channel)


    # 对梯度幅值进行二值化
    threshold_value = 100  # 阈值
    _, thresholdedL_channel = cv2.threshold(l_channel, threshold_value, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresholdedL_channel", thresholdedL_channel)

    # 合并多个二值图像的并集
    result = cv2.bitwise_and(thresholdedL_channel, cv2.bitwise_or(thresholdedSobelx, thresholdedS_channel))
    return result

def fill_lane_poly(img, left_fit, right_fit):
    # 获取图像的行数
    y_max = img.shape[0]
    print(y_max)
    # 设置输出图像的大小，并将白色位置设为255
    out_img = np.dstack((img, img, img)) * 255
    # 在拟合曲线中获取左右车道线的像素位置
    left_points = [[left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2], y] for y in range(y_max)]
    right_points = [[right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2], y] for y in range(y_max - 1, -1, -1)]
    # 将左右车道的像素点进行合并
    line_points = np.vstack((left_points, right_points))
    # 根据左右车道线的像素位置绘制多边形
    cv2.fillPoly(out_img,([line_points]), (0, 255, 0))
    return out_img


# 精确定位车道线
def cal_line_param(binary_warped):
    # 1.确定左右车道线的位置
    # 统计直方图
    max = binary_warped.shape
    print(max)
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
    print(window_height)
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

    cv2.imshow("fit",binary_warped)
    # 3.用曲线拟合检测出的点,二次多项式拟合，返回的结果是系数
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    print(left_fit)
    print(right_fit)
    return  left_fit,right_fit



if __name__ == "__main__":
    img = cv2.imread("./test/test3.jpg")


    cv2.imshow('Frame', img)


    points = [[601, 448], [683, 448], [230, 717], [1097, 717]]
    M, M_inverse = cal_perspective_params(img, points)
    transform_img = img_perspect_transform(img, M)
    cv2.imshow("transform",transform_img)
    cv2.imshow("pipeline",pipeline(transform_img))
    # 拟合车道线
    left_fit, right_fit = cal_line_param(pipeline(transform_img))

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

    # 显示绘制了车道线的图像
    cv2.imshow('Lane Lines', image)


    # # 绘制安全区域
    # result = fill_lane_poly(transform_img, left_fit, right_fit)
    # cv2.imshow("re",result)



    while True:
            # 等待按键输入
            key = cv2.waitKey(1) & 0xFF

            # 如果按下 'q' 键，退出循环
            if key == ord('q'):
                break

        # 销毁窗口
    cv2.destroyAllWindows()






