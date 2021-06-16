import numpy as np
from numpy.linalg import det, lstsq, norm
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
import math
import cv2


def get_interest_points(image, feature_width,r_thre,min_d):
    '''
    Returns interest points for the input image 返回输入图片的兴趣点

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''
    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!
    # xs = np.zeros(1)
    # ys = np.zeros(1)
    window_size = 3
    offset = int(window_size / 2)
    k = 0.04
    image = cv2.GaussianBlur(image, (3, 3), 0)  # 先对图像做平滑，避免存在一些变化剧烈的值
    ix = filters.sobel_v(image)
    iy = filters.sobel_h(image)
    Ixx = ix * ix
    Ixy = ix * iy
    Iyy = iy * iy
    height = image.shape[0]
    width = image.shape[1]
    image_R = np.zeros_like(image)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            windowIxx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1]  # 小窗内所有点的Ixx
            windowIxy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIyy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1]

            windowIxx = cv2.GaussianBlur(windowIxx, (window_size, window_size),
                                         0)  # sigma等于0的话会根据ksize自动计算两个方向的sigma 5x5的窗口计算出sigma=1.1
            windowIxy = cv2.GaussianBlur(windowIxy, (window_size, window_size), 0)
            windowIyy = cv2.GaussianBlur(windowIyy, (window_size, window_size), 0)

            Sxx = windowIxx.sum()  # 该窗口内所有Ixx之和即为最终该点的Ixx
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            r = det - k * (trace ** 2)
            image_R[y, x] = r
    thres = np.max(image_R)
    check1 = image_R > r_thre * thres  #r_thre越小，检测到的角点越多
    check2 = feature.peak_local_max(image_R, min_distance=min_d, indices=False)
    final = check1 & check2  # 满足条件的点为True
    pos = np.nonzero(final)
    ys = pos[0]
    xs = pos[1]
    # dst = cv2.cornerHarris(gray,2,3,0.04)
    return xs, ys


def get_features(image, x, y, feature_width):
    '''
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''
    # TODO: Your implementation here! See block comments and the project webpage for instructions
    # This is a placeholder - replace this with your features!
    features = np.zeros((0, 128))
    gauss_sigma = 1.5
    radius = int(np.round(3 * gauss_sigma))  # 4
    bins = 36
    # image=image[y,x] #根据y和x取出对应的坐标点
    # ix = filters.sobel_v(image)
    # print(ix.shape)
    # iy = filters.sobel_h(image)
    # mag = np.sqrt(ix ** 2 + iy ** 2)  # 求出每个点的梯度幅值
    # print(mag)
    # ori = np.arctan2(iy, ix)  # 求出每个点的梯度方向，不过是弧度值
    # print(ori)
    econs = -0.5 / (gauss_sigma ** 2)  # 高斯平滑指数
    hist = np.zeros((bins, image.shape[0], image.shape[1]))  # 维度为[36,h,w] 先求整张图的直方图

    bin_step = bins / 360.
    float_tolerance = 1e-7

    for j, i in zip(y, x):  # 对于每一组关键点 计算直方图，那么非关键点处的直方图值就为0
        for idy in range(-radius, radius + 1):
            for idx in range(-radius, radius + 1):
                if j + idy <= 0 or i + idx <= 0 or j + idy >= image.shape[0] - 1 or i + idx >= image.shape[
                    1] - 1:  # 该点超出了范围
                    continue
                new_y = j + idy
                new_x = i + idx

                dx = image[new_y, new_x + 1] - image[new_y, new_x - 1]
                dy = image[new_y - 1, new_x] - image[new_y + 1, new_x]

                now_mag = np.sqrt(dx * dx + dy * dy)
                now_ori = np.rad2deg(np.arctan2(dy, dx))  # 角度
                hist_index = int(round(now_ori * bin_step))
                # if bin < 0:  # 保证bin的范围在[0,35]之间
                #     bin += bins
                # elif bin >= bins:
                #     bin -= bins
                # print(bin)
                # 四舍五入之后再变成整型
                # bin = int(round(bin))
                # bin = bin % bins
                # if (bin == 36):
                #     bin = 0
                weight = np.exp((idx ** 2 + idy ** 2) * econs)  # 有值
                hist[hist_index % bins, j, i] += weight * now_mag  # 更新直方图


        # '用相邻bin的值对直方图做平滑'
        # for k in range(2):  # 平滑两次
        #     for n in range(bins):
        #         prev_idx = n - 1
        #         next_idx = n + 1
        #         if (n == 0):
        #             prev_idx = bins - 1
        #         if (n == bins - 1):
        #             next_idx = 0
        #
        #         hist[n, j, i] = 0.5 * hist[n, j, i] + (hist[prev_idx, j, i] + hist[next_idx, j, i]) * 0.25

        # 另一种插值方式
        for n in range(bins):
            hist[n, j, i] = (6 * hist[n, j, i] + 4 * (hist[n - 1, j, i] + hist[(n + 1) % bins, j, i]) + hist[
                n - 2, j, i] + hist[(n + 2) % bins, j, i]) / 16
    '抛物线插值'
    loc_angle = np.zeros((bins, image.shape[0], image.shape[1]))
    for j, i in zip(y, x):
        max_thre = np.max(hist[:, j, i], axis=0)  # 求当前关键点36个bin中的最大值 理论上是一个数
        thre = 0.8 * max_thre  # 辅方向要大于这个阈值且是局部最大值才进行插值
        for n in range(bins):  # 可能某关键点存在多个方向

            prev_value = hist[(n - 1) % bins, j, i]
            next_value = hist[(n + 1) % bins, j, i]

            if (hist[n, j, i] > prev_value and hist[n, j, i] > next_value and
                    hist[n, j, i] >= thre):  # 是局部最大值？
                new_bin = n + 0.5 * (prev_value - next_value) / (
                        prev_value + next_value - 2 * hist[n, j, i])
                new_bin = new_bin % bins

                new_angle = 360. - (new_bin / bins) * 360.  # 转换成角度不是弧度
                if abs(new_angle - 360.) < float_tolerance:
                    new_angle = 0

                new_bin = int(round(new_bin))  # 四舍五入再取整
                new_bin = new_bin % bins
                loc_angle[new_bin, j, i] = new_angle  # j,i这个点可能有很多个angle

    d = 4
    scale_multiplier = 3
    hist_w = scale_multiplier * 0.5 * (feature_width / 4)  #3 每个小cell4个像素
    radius = int(round(hist_w * (d + 1) * np.sqrt(2) * 0.5))  # 10
    exp_sigma = -0.5 / ((0.5 * d) ** 2)

    bin_num = 8
    bin_step = bin_num / 360.
    hist = np.zeros((6, 6, 8))  #
    descriptor_max_value = 0.2
    # print("复制点前x的个数：",len(x)) #668
    # print("复制点前y的个数：", len(y)) #668

    for j, i in zip(y, x):
        count_angle = 0
        for angle in loc_angle[:, j, i]:  # 取出该点的所有角度 因为关键点多了 所以换成用最大值代替
            if angle == 0:
                continue
            count_angle += 1
            cos = np.cos(np.deg2rad(angle))  # 求出主方向的sin和cos值
            sin = np.sin(np.deg2rad(angle))

            for idy in range(-radius, radius + 1):
                for idx in range(-radius, radius + 1):
                    x_rot = cos * idx - sin * idy
                    y_rot = sin * idx + cos * idy
                    ybin = (y_rot / hist_w) + d / 2 - 0.5  # 属于哪个cell
                    xbin = (x_rot / hist_w) + d / 2 - 0.5
                    if (xbin <= -1 or xbin >= 4 or ybin <= -1 or ybin >= 4):
                        continue
                    nowx = i + idx
                    nowy = j + idy

                    if nowy <= 0 or nowx <= 0 or nowy >= image.shape[0] - 1 or nowx >= image.shape[
                        1] - 1:  # 该点超出了范围
                        continue

                    dx = image[nowy, nowx + 1] - image[nowy, nowx - 1]
                    dy = image[nowy - 1, nowx] - image[nowy + 1, nowx]
                    now_mag = np.sqrt(dx * dx + dy * dy)
                    now_ori = np.rad2deg(np.arctan2(dy, dx)) % 360  # 角度值

                    weight = np.exp(((y_rot/ hist_w) ** 2 + (x_rot/hist_w) ** 2) * exp_sigma)
                    weight = weight * now_mag
                    now_ori -= angle  # 将当前点的方向与关键点的方向相减，求得旋转后的方向

                    hist_bin = now_ori * bin_step  # 得到属于哪个bin 值应当在0-7之间

                    if hist_bin < 0:
                        hist_bin += bin_num
                    if hist_bin >= bin_num:
                        hist_bin -= bin_num

                    trilinear_interpolate(
                        xbin, ybin, hist_bin, weight, hist)

            # print('hist：',hist.shape)
            hist = hist[1:5, 1:5, :].reshape(1, 128)  # 裁掉两边多余的两行两列

            hist /= max(norm(hist), float_tolerance)  # 变成单位向量,归一化一次
            hist[hist > descriptor_max_value] = descriptor_max_value

            hist /= max(norm(hist), float_tolerance)  # 再归一化一次
            # hist = np.round(512 * hist)
            # # print(hist)
            # hist[hist < 0] = 0
            # hist[hist > 255] = 255
            # hist = hist / max(norm(hist), float_tolerance)

            features = np.concatenate((features, hist), axis=0)
            hist = np.zeros((6, 6, 8))  # 重置hist,对下一个关键点进行计算

            if (count_angle > 1):
                x_idx = np.where(x == i)[0]  # 当某个点有多个辅方向时，复制关键点,可能有多个相同的值，但要取出位于同一处的
                y_idx = np.where(y == j)[0]

                final = set(x_idx).intersection(set(y_idx))  # 求交集，因为个数可能不一致
                list_f = list(final)[0]  # 变成List类型，方便取出值

                x = np.insert(x, list_f + 1, i)
                y = np.insert(y, list_f + 1, j)
        # print("该点的angle数目：",count_angle)

    print(features.shape)  # 图1：(1043,128)维  图2：(1014,128维)
    return features, x, y


def trilinear_interpolate(xbin, ybin, hbin, weight, hist):
    ybinf = int(np.floor(ybin))  # 向下取整
    xbinf = int(np.floor(xbin))
    hbinf = int(np.floor(hbin))

    dx = xbin - xbinf  # 小数余项
    dy = ybin - ybinf
    do = hbin - hbinf
    '''
    一个关键点周围的四个cell和每个cell的两个bin的权重
    b1点的中心 [ybinf,xbinf]    对应的bin的索引为hbinf和hbinf+1
    b2点的中心 [ybinf,xbinf+1]
    b3点的中心 [ybinf+1,xbinf]
    b2点的中心 [ybinf+1,xbinf+1]   
    '''
    w_b1_o1 = weight * (1 - dx) * (1 - dy) * (1 - do)
    w_b1_o2 = weight * (1 - dx) * (1 - dy) * do

    w_b2_o1 = weight * dx * (1 - dy) * (1 - do)
    w_b2_o2 = weight * dx * (1 - dy) * do

    w_b3_o1 = weight * (1 - dx) * dy * (1 - do)
    w_b3_o2 = weight * (1 - dx) * dy * do

    w_b4_o1 = weight * dx * dy * (1 - do)
    w_b4_o2 = weight * dx * dy * do

    '因为计算的b1点可能是(-1,-1)点，所以要加1'
    ybinf += 1
    xbinf += 1

    hist[ybinf, xbinf, hbinf % 8] += w_b1_o1
    hist[ybinf, xbinf, (hbinf + 1) % 8] += w_b1_o2

    hist[ybinf, xbinf + 1, hbinf % 8] += w_b2_o1
    hist[ybinf, xbinf + 1, (hbinf + 1) % 8] += w_b2_o2

    hist[ybinf + 1, xbinf, hbinf % 8] += w_b3_o1
    hist[ybinf + 1, xbinf, (hbinf + 1) % 8] += w_b3_o2

    hist[ybinf + 1, xbinf + 1, hbinf % 8] += w_b4_o1
    hist[ybinf + 1, xbinf + 1, (hbinf + 1) % 8] += w_b4_o2


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR. 设计一个置信度，对于距离相似的特征点，NNDR测试将返回一个接近1的数字

    This function does not need to be symmetric (e.g., it can produce 不需要使用对称的匹配
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''
    # TODO: Your implementation here! See block comments and the project webpage for instructions
    # These are placeholders - replace with your matches and confidences!
    # matches = np.zeros((1, 2))
    # confidences = np.zeros(1)
    # print(im1_features)
    distance_matrix = EuclideanDistance(im1_features, im2_features)
    # print(distance_matrix)  # 1043,1014

    min_first = np.min(distance_matrix, axis=1)  # 理论上为(n,) 求行最小值
    min_first = min_first.reshape(min_first.shape[0], 1)  # 变成[n,1]维
    # print(min_first.shape)
    min_first_index = np.argmin(distance_matrix, axis=1)  # 求出最小值的索引，理论上也为[n,] 与图1中的点距离最小的图2中点的索引
    # print(min_first_index)
    # print('min_first_index 的维度：',min_first_index.shape) #（1043,）

    for i in range(0, distance_matrix.shape[0]):  # 根据最小值将数组中等于最小值的元素替换成 无穷大
        change_ixs = np.where(distance_matrix[i] == min_first[i])
        distance_matrix[i][change_ixs] = np.inf
    min_second = np.min(distance_matrix, axis=1)  # 求次最小值
    min_second = min_second.reshape(min_second.shape[0], 1)  # 变成[n,1]维
    min_second_index = np.argmin(distance_matrix, axis=1)

    # print(min_first)
    # print(min_second)
    ratio = min_first / min_second
    # print(ratio)
    # print('ratio维度：',ratio.shape)  (1043, 1)
    ratio_threshold = 0.6
    match_bool = ratio < ratio_threshold  # (711,1) 值为True或False
    match_1_index = np.nonzero(match_bool)[0]  # 取出值为True的图1中的匹配点索引 维度为(144,1)
    # print(match_1_index)
    print('匹配点个数', len(match_1_index))
    match_2_index = min_first_index[match_1_index]

    confidences = 1-ratio[match_1_index] #本来符合条件的值是小于0.5的，和1相减后就是一个接近1的数字

    match_1_index = match_1_index.reshape(match_1_index.shape[0], 1)  # 维度为(144,1)
    match_2_index = match_2_index.reshape(match_2_index.shape[0], 1)
    matches = np.vstack((np.transpose(match_1_index), np.transpose(match_2_index)))  # 求转置前维度为(2,144)
    matches = np.transpose(matches)  # 维度为(144,2)
    confidences = confidences.reshape(confidences.shape[0], )
    print('confidence维度：', confidences.shape)  # (144,)
    print('matches维度：', matches.shape)

    return matches, confidences


def EuclideanDistance(x, y):
    """
    get the Euclidean Distance between two matrix
    (x-y)^2 = x^2 + y^2 - 2xy
    :param x:
    :param y:
    :return:
    """
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dis = x2 + y2 - 2 * xy
    return dis
