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
                fnd=True
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


def upper_contour_features(text_lines):
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
    for mat in text_lines:
        for j in range(0, mat.shape[1]):
            for i in range(0, mat.shape[0]):
                if mat[i][j] == 0:
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
        slant, mse = contour_slant_mse(x, y)
        max_freq, max_slope = local_maxima(x, y)
        min_freq, min_slope = local_minima(x, y)
        slant_list.append(slant)
        mse_list.append(mse)
        max_freq_list.append(max_freq)
        max_slope_list.append(max_slope)
        min_freq_list.append(min_freq)
        min_slope_list.append(min_slope)

    return sum(slant_list) / len(slant_list), sum(mse_list) / len(mse_list), sum(max_freq_list) / len(max_slope_list), sum(max_slope_list) / len(max_slope_list), sum(min_freq_list) / len(min_freq_list), sum(min_slope_list) / len(min_slope_list)


def lower_contour_features(text_lines):
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
    for mat in text_lines:
        for j in range(0, mat.shape[1]):
            for i in range(mat.shape[0]-1, -1, -1):
                if mat[i][j] == 0:
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
        slant, mse = contour_slant_mse(x, y)
        max_freq, max_slope = local_maxima(x, y)
        min_freq, min_slope = local_minima(x, y)
        slant_list.append(slant)
        mse_list.append(mse)
        max_freq_list.append(max_freq)
        max_slope_list.append(max_slope)
        min_freq_list.append(min_freq)
        min_slope_list.append(min_slope)
    return sum(slant_list) / len(slant_list), sum(mse_list) / len(mse_list), sum(max_freq_list) / len(max_slope_list), sum(max_slope_list) / len(max_slope_list), sum(min_freq_list) / len(min_freq_list), sum(min_slope_list) / len(min_slope_list)


def removing_upper_lower(img):
    """
    because IAM database has 2 upper and 1 lower lines that need to be removed
    :param img: binary image
    :return: image without unnecessary frames
    """
    img_portion = img[:, 500:2000]  # removing the footer and the left side and the first horizontal line of the page
    margin = 14
    r = 0
    # ***** this comments are for vectorization trial but it wasn't success so I loop over the rows of the image
    # new_img = np.reshape(img_portion, (len(img_portion)//14, 14, len(img_portion[0])))
    # hist = np.amin(new_img, axis=1).sum(axis=1)
    # print(np.argwhere(hist==0)*14)
    hor_lines = []
    while r < len(img_portion) - margin:
        if np.amin(img_portion[r : r + margin], axis=0).sum() == 0:
            hor_lines.append(r)
            r += 100
        r += 1
    if len(hor_lines) >= 3:
        return img[hor_lines[1] + 30:hor_lines[2], 70:]
    else:
        print("ERROR")
        img_print(img_portion[hor_lines[0]:])
        img_print(img_portion[hor_lines[1]:])


def split_lines(img):
    """
    splits a page into text lines
    :param img: binary image
    :return: returns a vector of text line images
    """
    # this array contains summation of all *black* pixels on each row of the image
    row_hist = np.sum(img<255, axis=1)
    # threshold for rows contains black pixel less than 25 (keeps only ones with higher black pixels that 25)
    is_lines = row_hist>100
    lines = []
    i = 0
    while i < len(is_lines):
        if is_lines[i]:
            begin_row = i
            lower_bound = max(begin_row - 50, 0)
            while i < len(is_lines) and is_lines[i]:
                i+=1
            upper_bound = min(i + 50, len(is_lines)-1)
            if i - begin_row > 20: # threshold for # of rows to be higher than 20 row
                lines.append(img[lower_bound:upper_bound, :])
        i += 1
    return lines


if __name__ == '__main__':
    #directory = "formsA-D/"
    file_name = "1.PNG"
    #for file_name in listdir(directory):
    img = load_binary_img(file_name)
    img = removing_upper_lower(img)
    lines = split_lines(img)

    for indx, line in enumerate(lines):
        write_img(line, "text_lines/" + file_name[:-4], indx)

    upper_contour_features(lines)
    lower_contour_features(lines)
    '''''''''
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    s.append(arr)
    for mat in s:
        print(type(mat))
        for i in range(0, mat.shape[0]):
            for j in range(0, mat.shape[1]):
                print(mat[i][j])*/
    '''''''''


