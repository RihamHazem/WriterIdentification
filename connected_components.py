import cv2
import os
from misc import write_img, img_print
from split_page import *
from component import *


def get_words_lens_avg_dist(components):
    """
    classify components into words, get average width of components and components' width
    :param components: list of Component
    :return: generated words, average width of component and list of components' width
    """
    i = 1
    dist = 0
    left_most = components[0].left_most
    top_most = components[0].top_most
    right_most = components[0].get_right_most()
    bottom_most = components[0].get_bottom_most()
    co_area = components[0].co_area

    words = []  # 2D array contains array of components represents the word
    comp_lens = [components[0].box_width]  # list of width of components
    t = 25  # threshold of distance between words
    while i < len(components):
        comp_lens.append(components[i].box_width)
        co_dist = components[i].left_most - components[i - 1].get_right_most()
        dist += co_dist
        if co_dist <= t:
            top_most = min(top_most, components[i].top_most)
            right_most = max(right_most, components[i].get_right_most())
            bottom_most = max(bottom_most, components[i].get_bottom_most())
            co_area += components[i].co_area
        else:
            words.append(Component(left_most, top_most, right_most - left_most, bottom_most - top_most, co_area))

            left_most = components[i].left_most
            top_most = components[i].top_most
            right_most = components[i].get_right_most()
            bottom_most = components[i].get_bottom_most()
            co_area = components[i].co_area

        i += 1
    words.append(Component(left_most, top_most, right_most - left_most, bottom_most - top_most, co_area))
    dist /= len(components) - 1

    return dist, words, np.array(comp_lens)


def avg_word_dist_and_width(words):
    avg_width = words[0].box_width
    avg_dist = 0
    i = 1
    while i < len(words):
        avg_width += words[i].box_width
        avg_dist += words[i].left_most - words[i - 1].get_right_most()
        i += 1
    return avg_width / len(words), avg_dist / (len(words) - 1)


def print_words(words, line):
    print(words)
    print("============================================")
    for comp in words:
        cv2.rectangle(line, (comp.left_most, comp.top_most), (comp.get_right_most(), comp.get_bottom_most()), (0, 0, 0),
                      2)
    img_print(line)


def avg_black_to_white(components, line):
    """
    calculates average black to white transition of each component
    :param components:
    :return: avg #
    """
    avg_b_w = 0
    cnt = 0
    for component in components:
        bin_img = line[:, component.left_most:component.get_right_most()]
        diff_img = np.diff(bin_img.astype(int))

        unique, counts = np.unique(diff_img, return_counts=True)
        counts = dict(zip(unique, counts))
        if 255 in counts:
            avg_b_w += counts[255]
        cnt += 1
    return avg_b_w / cnt


def line_features(components, line):
    components.sort(key=lambda x: x.left_most)
    """
        features of one line:
            feature1 => avg dist between components     feature2 => avg dist between words
            feature3 => avg lens of word                feature4 => avg lens of component
            feature5 => median lens of component        feature6 => std lens of component       
            feature7 => avg black to white transition over all components
            feature8 => average length of white portions in most row that has black white transition or white black (basic features)
        :return array of the above features
    """
    features = []
    feature1, words, comp_lens = get_words_lens_avg_dist(components)

    feature2, feature3 = avg_word_dist_and_width(words)

    features.append(feature1)
    features.append(feature2)
    features.append(feature3)
    features.append(np.mean(comp_lens))
    features.append(np.median(comp_lens))
    features.append(np.std(comp_lens))

    features.append(avg_black_to_white(components, line))
    return features
