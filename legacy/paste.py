import cv2
import os
import argparse
import copy
import itertools
import numpy as np


def euclidean_distance(_vector1, _vector2):
    """
    :param _vector1:  Input vector
    :param _vector2:  Input vector
    :return:  Squared value of euclidean distance.
    """
    assert _vector1.shape == _vector2.shape
    return np.sqrt(np.sum((_vector1 - _vector2) ** 2))


def load_cut_images(_dir_path):
    """
    :param _dir_path:  The directory path where the cropped images are stored.
    :return:  All jpg format images in directory path.
    """
    result = list()
    for file_name in os.listdir(_dir_path):
        split_name, ext = os.path.splitext(file_name)
        if ext == '.jpg':
            result.append(cv2.imread(os.path.join(_dir_path, file_name)))
            # result.append(cv2.cvtColor(cv2.imread(os.path.join(_dir_path, file_name)), cv2.COLOR_RGB2HSV))
    return result


def paste_row_wise(_base_img, _cand_img):
    top_measure = 0
    bottom_measure = 0

    for i in range(_base_img.shape[1]):
        top_measure += euclidean_distance(_base_img[0, i, :], _cand_img[-1, i, :])
        bottom_measure += euclidean_distance(_base_img[-1, i, :], _cand_img[0, i, :])

    top_measure = (-1 * top_measure / _base_img.shape[1])
    bottom_measure = (-1 * bottom_measure / _base_img.shape[1])
    result_state = [top_measure, 'top'] if top_measure > bottom_measure else [bottom_measure, 'bottom']

    stacked_img = None
    if result_state[1] == 'top':
        stacked_img = np.vstack((_cand_img, _base_img))
    elif result_state[1] == 'bottom':
        stacked_img = np.vstack((_base_img, _cand_img))

    return stacked_img, result_state

"""
def paste_row_wise(_base_img, _cand_img):

    top_measure = 0
    bottom_measure = 0

    for i in range(_base_img.shape[1]):
        top_measure += euclidean_distance(_base_img[0, i, :], _cand_img[-1, i, :])
        bottom_measure += euclidean_distance(_base_img[-1, i, :], _cand_img[0, i, :])

    top_measure = top_measure * (-1)
    bottom_measure = bottom_measure * (-1)
    result_state = [top_measure, 'top'] if top_measure > bottom_measure else [bottom_measure, 'bottom']

    stacked_img = None
    if result_state[1] == 'top':
        stacked_img = np.vstack((_cand_img, _base_img))
    elif result_state[1] == 'bottom':
        stacked_img = np.vstack((_base_img, _cand_img))

    return stacked_img, result_state
"""


def paste_col_wise(_base_img, _cand_img):

    right_measure = 0
    left_measure = 0
    """
    for i in range(3):
        right_measure += euclidean_distance(_base_img[:, -1, i], _cand_img[:, 0, i])
        left_measure += euclidean_distance(_base_img[:, 0, i], _cand_img[:, -1, i])
    """
    for i in range(_base_img.shape[0]):
        right_measure += euclidean_distance(_base_img[i, -1, :], _cand_img[i, 0, :])
        left_measure += euclidean_distance(_base_img[i, 0, :], _cand_img[i, -1, :])

    right_measure = (-1 * right_measure / _base_img.shape[0])
    left_measure = (-1 * left_measure / _base_img.shape[0])
    result_state = [right_measure, 'right'] if right_measure > left_measure else [left_measure, 'left']

    stacked_img = None
    if result_state[1] == 'right':
        stacked_img = np.hstack((_base_img, _cand_img))
    elif result_state[1] == 'left':
        stacked_img = np.hstack((_cand_img, _base_img))

    return stacked_img, result_state


def mirroring(_img):
    return cv2.flip(_img, 1)


def flipping(_img):
    return cv2.flip(_img, 0)


def unrotate(_img):
    return cv2.rotate(_img, cv2.ROTATE_90_CLOCKWISE)


def rotate(_img):
    return cv2.rotate(_img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def index_combination(_index_list):
    """
    :param _index_list: input list.
    :return: Returns all combinations of list elements.
    """
    combs = list()
    for i in range(len(_index_list)+1):
        els = [list(x) for x in itertools.combinations(_index_list, i)]
        combs.extend(els)

    return combs


def transform(_image, _decode_list, _func_idx):
    """
    :param _image: Input RGB or Gray image.
    :return: Decoded (mirroring, flipping, rotate) image.
    """
    for _idx in _func_idx:
        decoded_image = copy.deepcopy(_image)
        for _i in range(len(_idx) - 1, -1, -1):
            decoded_image = _decode_list[_idx[_i]](decoded_image)
        yield decoded_image


if __name__ == "__main__":

    row = 3
    col = 3
    dir_path = "./cut_image"
    loaded_images = load_cut_images(dir_path)

    encode_list = [mirroring, flipping, unrotate, rotate]
    decode_list = [mirroring, flipping, unrotate, rotate]
    func_idx = index_combination(range(len(decode_list)))

    row_wise_images = list()
    n = 0
    standard_shape = None
    for r in range(row):
        # state_dict = dict()
        best_measure = None
        best_measure_img = None
        best_idx = None
        base_img_list = list()
        base_img = loaded_images.pop(0)

        if not standard_shape:
            standard_shape = base_img.shape
            base_img_list.append(base_img)
        else:
            for base_decode in transform(base_img, decode_list, func_idx):
                if standard_shape == base_decode.shape:
                    base_img_list.append(base_decode)

        for final_base_img in base_img_list:
            for idx, candidate_img in enumerate(loaded_images):
                for decode_img in transform(candidate_img, encode_list, func_idx):
                    # cv2.imwrite("./decode_%d.jpg" % n, decode_img)
                    # n += 1
                    if decode_img.shape[1] != final_base_img.shape[1]:
                        continue
                    pasted_img, pasted_state = paste_row_wise(final_base_img, decode_img)

                    # print("%d_%d_%d.jpg" % (r, idx, n))
                    # print("%d" % n, ":", pasted_state[0])
                    # cv2.imwrite("%d.jpg" % n, pasted_img)
                    # n += 1

                    if not best_measure or pasted_state[0] > best_measure:
                        best_measure_img = pasted_img
                        best_measure = pasted_state[0]
                        best_idx = idx
                        # print(n, ":", best_measure)
                        # cv2.imwrite("%d_%d_%d.jpg" % (r, idx, n), best_measure_img)
        loaded_images.pop(best_idx)
        row_wise_images.append(best_measure_img)
        # print("*"*50)

    for i, img in enumerate(row_wise_images):
        cv2.imwrite("./%d.jpg" % i, img)

    # col_wise_images = list()
    final_image = None
    # m = 0
    for c in range(col - 1):
        # state_dict = dict()
        best_measure = None
        best_measure_img = None
        best_idx = None
        base_img = row_wise_images.pop(0)
        for idx, candidate_img in enumerate(row_wise_images):
            for decode_img in transform(candidate_img, encode_list, func_idx):
                if decode_img.shape[0] != base_img.shape[0]:
                    continue

                pasted_img, pasted_state = paste_col_wise(base_img, decode_img)

                # print(m, ":", pasted_state[0])
                # cv2.imwrite("./_%d.jpg" %(m), pasted_img)
                # m += 1
                if not best_measure or pasted_state[0] > best_measure:
                    final_image = pasted_img
                    best_measure = pasted_state[0]
                    best_idx = idx
        row_wise_images.pop(best_idx)

    cv2.imwrite("./final_prev.jpg", final_image)


