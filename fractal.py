import cv2
import math
import skimage.morphology as morph
import matplotlib.pyplot as plt
import numpy as np


def getfractalftrs(textline):
    txtln = np.zeros(textline.shape)
    mask = textline > 0
    txtln[mask] = 1
    lnd, lnad = diskdilation(textline)
    f1 = getslope(lnd, lnad)
    return f1


def disk_kernel(r):
    kernel = np.zeros((2 * r + 1, 2 * r + 1))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    mask = x ** 2 + y ** 2 <= r ** 2
    kernel[mask] = 1

    return kernel


def elps_kernel(a, b, theta):
    kernel = np.zeros((2 * a + 1, 2 * b + 1))
    cv2.ellipse(kernel, (a, b), (a, b), 0, 0, theta, 1, -1)
    return kernel


def diskdilation(txtln):
    x = []
    y = []
    l, w = txtln.shape
    maxd = min(l // 4, w // 4)
    for dmtr in range(1, maxd):
        kernel = disk_kernel(dmtr // 2)
        dltline = morph.binary_dilation(txtln, selem=kernel)   # line after applying dilation
        a = np.count_nonzero(dltline == 1)  # area after dilation:# of ones
        lnd = np.log(dmtr)
        x.append(lnd)
        y.append(np.log(a) - lnd)
    return x, y


# plot the three lines each with different colors
def pltgraph(lins):
    plt.xlabel("log d")
    plt.ylabel("logA(d)-log(d)")
    clrs = ['-r', '-b', 'grey']
    for i, l in enumerate(lins):
        plt.plot(l[0], l[1], clrs[i])
    plt.show()


def fitline(x, y, pts,min_err):
    lins = []
    slp = []
    err = 0
    for i in range(3):
        par = np.polyfit(x[pts[i]:pts[i + 1]], y[pts[i]:pts[i + 1]], 1, full=True)  # find the best fit line
        slope = par[0][0]
        intercept = par[0][1]

        xd = [min(x[pts[i]:pts[i + 1]+1]), max(x[pts[i]:pts[i + 1]+1])]
        yd = [slope * xx + intercept for xx in xd]
        lins.append([xd, yd])

        # computing error for this line
        err = err + 1/len(x)*(sum(
            [(slope * xx + intercept - yy)**2 for xx, yy in zip(x[pts[i]:pts[i + 1]], y[pts[i]:pts[i + 1]])]))
        slp.append(slope)
        if min_err<err:
            break
    return lins, err, slp

def getslope(x, y):
    # sorting according to X
    reorder = sorted(range(len(x)), key=lambda ii: x[ii])
    x = [x[ii] for ii in reorder]
    y = [y[ii] for ii in reorder]

    # make the scatter plot
    plt.scatter(x, y, s=30, alpha=0.8, marker='o', cmap="BuPu")
    #plt.show()
    # search for the endpoints
    err = 100000  # max error
    l_slope = []  # the final slopes of the desired line
    zal = []  # the desired line
    pts = [0, 0, 0, len(x)]  # start and end points for the three lines
    rng=math.floor(len(x)*0.15)
    tholth=math.floor(len(x)*0.3)

    for p1 in range(max(tholth,(len(x)//3)-rng), min(len(x)-(2*tholth),(len(x)//3)+rng)):
     for p2 in range(p1+tholth, len(x)-rng):
            pts[1] = p1
            pts[2] = p2
            lines, lines_err, slopes = fitline(x, y, pts,err)
            if lines_err < err:
                l_slope = slopes
                zal = lines
                err=lines_err
    #pltgraph(zal)
    return l_slope






