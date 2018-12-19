from cv2 import *
from os import path, makedirs
import shutil


def delete_dir(directory):
    if path.exists(directory):
        shutil.rmtree(directory)


def make_dirs(directory):
    if not path.exists(directory):
        makedirs(directory)


def load_binary_img(file_path):
    img = imread(file_path, 0)
    _, img = threshold(img, 200, 255, THRESH_BINARY)
    return img


def write_img(img, directory, global_cnt):
    make_dirs(directory)
    imwrite(directory + "/" + str(global_cnt) + ".png", img)


def img_print(img):
    namedWindow('image', WINDOW_NORMAL)
    imshow('image', img)
    waitKey(0)
    destroyAllWindows()