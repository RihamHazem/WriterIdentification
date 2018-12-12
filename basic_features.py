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

    bl = zaline.shape[0]  # bottom line
    tl = 0  # the top line
    ul = 1
    lb = 1

    m = math.floor(zaline.shape[0] * 0.2)
    h = sum(hist[m:m+5])
    err = sum((h - hist) ** 2)

    for u in range(m+1, zaline.shape[0] - m - 5):
        for l in range(u+4, zaline.shape[0] - m):
            h = sum(hist[u:l+1])
            e = sum((h - hist) ** 2)
            if err >e:
                ul = u
                lb = l
                err = e
    return tl, ul, lb, bl


def basic_ftrs_one(zaline):
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

    return np.array([f1, f2, f3, f4, f5 , f6])















