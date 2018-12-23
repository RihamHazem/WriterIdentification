import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def historgram(zaline):
    zahist = []
    for i in range(zaline.shape[0]):
        zahist.append(zaline[i].sum(axis=0))
    return zahist


def getboundaries(zaline):
    # get the histogram
    hist = historgram(zaline)
    m = math.floor(zaline.shape[0] * 0.3)

    bl = zaline.shape[0]  # bottom line
    tl = 0  # the top line
    ul = m
    lb = 2*m
    h = sum(hist[m:(2*m)+1])
    err = sum((h - hist) ** 2)

    for u in range(m+1, zaline.shape[0] - 2*m):
        for l in range(u+m, zaline.shape[0] - m):
            h = sum(hist[u:l+1])
            e = sum((h - hist) ** 2)
            if err >e:
                ul = u
                lb = l
                err = e
    return tl, ul, lb, bl


def basic_ftrs(zaline):
    # convert to binary image
    line_bw = cv2.threshold(zaline, 0, 1, cv2.THRESH_BINARY)[1]

    tl, ul, lb, bl = getboundaries(line_bw)
    f = []
    f1 = abs(tl-ul)
    f2 = abs(ul-lb)
    f3 = abs(lb-bl)
    f4 = f1/f2
    f5 = f1/f3
    f6 = f2/f3
    f7 = count_white_portion( most_row_BW_WB(zaline) )
    f8 = f7/f2
    return [f1, f2, f3, f4, f5 , f6, f7, f8]


# from basic features
def most_row_BW_WB(line):
    """
    calculates the index of row that has highest black-white and white-black transitions
    :param line: image of the text line
    :return: most row black to white and white to black
    """
    diff_img = np.diff(line.astype(int))
    row_number = -1
    mx_cnt = 0
    i = 0

    for diff_row in diff_img:
        unique, counts = np.unique(diff_row, return_counts=True)
        counts = dict(zip(unique, counts))
        num_255 = 0
        num_neg_255 = 0
        if 255 in counts:
            num_255 = counts[255]
        if -255 in counts:
            num_neg_255 = counts[-255]
        if mx_cnt < num_255 + num_neg_255:
            mx_cnt = num_255 + num_neg_255
            row_number = i
        i += 1
    return line[ row_number ]


# from basic features
def count_white_portion(row):
    """
    calculates the white portions' size and returns the median one
    :param row: row of the text line (the one that has highest number of transitions (B-W & W-B)
    :return: the median size of white portions od given row
    """
    first_black = False
    last_black = False
    idx = 0
    d = []
    while idx < len(row):
        if row[idx] < 255:
            first_black = True
            last_black = True
        cnt = 0
        while first_black and idx < len(row) and row[idx] == 255:
            cnt += 1  # counting the width of continuous white pixels
            last_black = False
            idx += 1
        if cnt == 0:
            idx += 1
        else:
            d.append(cnt)
    if last_black:
        d.pop()
    return np.median(d)

