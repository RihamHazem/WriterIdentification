import numpy as np
from os import listdir

from misc import img_print, write_img, load_binary_img


def removing_upper_lower(img):
    """
    because IAM database has 2 upper and 1 lower lines that need to be removed
    :param img: binary image
    :return: image without unnecessary frames
    """
    img_portion = img[:, 600:1900]  # removing the footer and the left side and the first horizontal line of the page

    margin = 20
    r = 0
    # ***** this comments are for vectorization trial but it wasn't success so I loop over the rows of the image
    # new_img = np.reshape(img_portion, (len(img_portion)//14, 14, len(img_portion[0])))
    # hist = np.amin(new_img, axis=1).sum(axis=1)
    # print(np.argwhere(hist==0)*14)
    hor_lines = []
    while r < len(img_portion) - margin:
        if np.amin(img_portion[r: r + margin], axis=0).sum() == 0:
            hor_lines.append(r)
            r += 100
        r += 1
    if len(hor_lines) >= 3:
        return img[hor_lines[1] + 30:hor_lines[2], 70:]
    else:
        print("ERROR", len(hor_lines))
        img_print(img)
        if len(hor_lines)>0:
            img_print(img_portion[hor_lines[0]:])
        if len(hor_lines)>1:
            img_print(img_portion[hor_lines[1]:])


def split_lines(img):
    """
    splits a page into text lines
    :param img: binary image
    :return: returns a vector of text line images
    """
    # this array contains summation of all *black* pixels on each row of the image
    row_hist = np.sum(img < 255, axis=1)
    # threshold for rows contains black pixel less than 25 (keeps only ones with higher black pixels that 25)
    is_lines = row_hist > 5
    lines = []
    i = 0
    while i < len(is_lines):
        if is_lines[i]:
            begin_row = i
            lower_bound = max(begin_row - 5, 0)
            while i < len(is_lines) and is_lines[i]:
                i += 1
            upper_bound = min(i + 5, len(is_lines) - 1)
            if i - begin_row > 50:  # threshold for # of rows to be higher than 20 row
                lines.append(img[lower_bound:upper_bound, :])
        i += 1
    return lines


if __name__ == '__main__':
    directory = "data"
    cnt = 1
    for test_case in listdir(directory):
        for writer_id in listdir(directory + "/" + test_case):
            if "test.PNG" in writer_id:
                continue
            for file_name in listdir(directory + "/" + test_case + "/" + writer_id):
                img = load_binary_img(directory + "/" + test_case + "/" + writer_id + "/" + file_name)
                img = removing_upper_lower(img)
                # img_print(img)
                lines = split_lines(img)

                for indx, line in enumerate(lines):
                    write_img(line, "lines_data/", cnt)
                    cnt += 1
