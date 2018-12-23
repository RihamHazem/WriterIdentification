import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from misc import img_print, write_img, load_binary_img


def contour_slant_mse(x, y):
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    y_pred = regr.predict(x)
    return regr.coef_[0], mean_squared_error(y, y_pred)


def av_slope(s):
    ret = 0.0
    for i in range(0, len(s)):
        ret += s[i]
    return ret / len(s)


def local_maxima(x, y):
    cnt = 0
    slopes = []
    for i in range(20, len(y)-21):
        fnd = False
        x_left_points = []
        x_right_points = []
        y_left_points = []
        y_right_points = []
        for j in range(i-20, i):
            x_left_points.append(x[j])
            y_left_points.append(y[j])
            if y[j] > y[i]:
                fnd = True
                break
        for j in range(i+1, i+21):
            x_right_points.append(x[j])
            y_right_points.append(y[j])
            if y[j] > y[i]:
                fnd = True
                break
        if not fnd:
            cnt += 1
            slope, _ = contour_slant_mse(x_left_points, y_left_points)
            slopes.append(slope)
            slope, _ = contour_slant_mse(x_right_points, y_right_points)
            slopes.append(slope)

    return cnt / len(y), av_slope(slopes)


def local_minima(x, y):
    cnt = 0
    slopes = []
    for i in range(20, len(y)-21):
        fnd = False
        x_left_points = []
        x_right_points = []
        y_left_points = []
        y_right_points = []
        for j in range(i-20, i):
            x_left_points.append(x[j])
            y_left_points.append(y[j])
            if y[j] < y[i]:
                fnd=True
                break
        for j in range(i+1, i+21):
            x_right_points.append(x[j])
            y_right_points.append(y[j])
            if y[j] < y[i]:
                fnd = True
                break
        #print(x_left_points)
        #print(y_left_points)
        #print(x_right_points)
        #print(y_right_points)
        if not fnd:
            cnt += 1
            slope, _ = contour_slant_mse(x_left_points, y_left_points)
            slopes.append(slope)
            slope, _ = contour_slant_mse(x_right_points, y_right_points)
            slopes.append(slope)

    return cnt / len(y), av_slope(slopes)


def upper_contour_features(text_line):
    y = []
    x = []
    ref = 0
    last = -1
    last_y = 0
    first = True
    slant_list = []
    mse_list = []
    max_freq_list= []
    max_slope_list = []
    min_freq_list = []
    min_slope_list = []
    for j in range(0, text_line.shape[1],1):
        for i in range(0, text_line.shape[0]):
            if text_line[i][j] == 0:
                if first:
                    ref = 0
                    first = False
                    last_y = i
                if last_y < i:
                    ref -= 1
                    y.append(ref)
                elif last_y > i:
                    ref += 1
                    y.append(ref)
                else:
                    y.append(ref)
                last_y = i
                ls = [last + 1]
                x.append(ls)
                last += 1
                break
    if len(x) == 0:
        return []
    slant, mse = contour_slant_mse(x, y)
    max_freq, max_slope = local_maxima(x, y)
    min_freq, min_slope = local_minima(x, y)
    slant_list.append(slant)
    mse_list.append(mse)
    max_freq_list.append(max_freq)
    max_slope_list.append(max_slope)
    min_freq_list.append(min_freq)
    min_slope_list.append(min_slope)

    return [sum(slant_list) / len(slant_list), sum(mse_list) / len(mse_list), sum(max_freq_list) / len(max_slope_list), sum(max_slope_list) / len(max_slope_list), sum(min_freq_list) / len(min_freq_list), sum(min_slope_list) / len(min_slope_list)]


def lower_contour_features(text_line):
    y = []
    x = []
    ref = -1
    last = -1
    first = True
    last_y = 0
    slant_list = []
    mse_list = []
    max_freq_list = []
    max_slope_list = []
    min_freq_list = []
    min_slope_list = []
    for j in range(0, text_line.shape[1]):
        for i in range(text_line.shape[0]-1, -1, -1):
            if text_line[i][j] == 0:
                if first:
                    ref = i
                    first = False
                if last_y < i:
                    ref -= 1
                    y.append(ref)
                elif last_y > i:
                    ref += 1
                    y.append(ref)
                else:
                    y.append(ref)
                ls = [last+1]
                x.append(ls)
                last += 1
                last_y = i
                break
    if len(x) == 0:
        return []
    slant, mse = contour_slant_mse(x, y)
    max_freq, max_slope = local_maxima(x, y)
    min_freq, min_slope = local_minima(x, y)
    slant_list.append(slant)
    mse_list.append(mse)
    max_freq_list.append(max_freq)
    max_slope_list.append(max_slope)
    min_freq_list.append(min_freq)
    min_slope_list.append(min_slope)
    return [sum(slant_list) / len(slant_list), sum(mse_list) / len(mse_list), sum(max_freq_list) / len(max_slope_list), sum(max_slope_list) / len(max_slope_list), sum(min_freq_list) / len(min_freq_list), sum(min_slope_list) / len(min_slope_list)]




