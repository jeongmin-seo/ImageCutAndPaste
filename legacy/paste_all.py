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

    top_measure = (-1 * top_measure /_base_img.shape[1])
    bottom_measure = (-1 * bottom_measure /_base_img.shape[1])
    result_state = [top_measure, 'top'] if top_measure > bottom_measure else [bottom_measure, 'bottom']

    stacked_img = None
    if result_state[1] == 'top':
        stacked_img = np.vstack((_cand_img, _base_img))
    elif result_state[1] == 'bottom':
        stacked_img = np.vstack((_base_img, _cand_img))

    return stacked_img, result_state


def paste_col_wise(_base_img, _cand_img):

    right_measure = 0
    left_measure = 0
    for i in range(3):
        right_measure += euclidean_distance(_base_img[:, -1, i], _cand_img[:, 0, i])
        left_measure += euclidean_distance(_base_img[:, 0, i], _cand_img[:, -1, i])

    right_measure = right_measure * (-1)
    left_measure = left_measure * (-1)
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

    n = 0

    base_img = loaded_images.pop(0)
    base_img_shape = base_img.shape
    # all_row_result = list()

    all_row_result = [[base_img, 0, loaded_images]]
    for _ in range(row - 1):
        # 반복해서 입력받을 수 있는 함수 작성

        tmp_all_row_result = list()
        for _, one_row_result in enumerate(all_row_result):
            for idx, candidate_img in enumerate(one_row_result[2]):
                # tmp_remain_images = copy.deepcopy(loaded_images)
                for decode_img_idx, decode_img in enumerate(transform(candidate_img, encode_list, func_idx)):
                    tmp_remain_images = copy.deepcopy(one_row_result[2])
                    if decode_img.shape[1] != one_row_result[0].shape[1]:
                        continue

                    pasted_img, pasted_state = paste_row_wise(one_row_result[0], decode_img)
                    tmp_measure = pasted_state[0] + one_row_result[1]
                    tmp_remain_images.pop(idx)
                    tmp_all_row_result.append([pasted_img, tmp_measure, tmp_remain_images])
        all_row_result = copy.deepcopy(tmp_all_row_result)

    all_col_result = list()
    # for _ in range(col - 1):
    for _, one_row_result in enumerate(all_row_result):
        remain_images = copy.deepcopy(one_row_result[2])
        new_base_img = remain_images.pop(0)
        tmp_all_col_result = list()

        for decoded_new_base_img in transform(new_base_img, encode_list, func_idx):
            if base_img_shape != decoded_new_base_img.shape:
                continue

            # 여기부터 동그라미 아닌 숫자 부분
            # 임의의 현재 상태 저장하고 이를 통해서 반복
            # col 반복
            col_state = dict()
            for c in range(col - 1):
                if not col_state:
                    all_middle_result = [[decoded_new_base_img, 0, remain_images]]
                else:
                    all_middle_result = []
                    for one_state in col_state[str(c-1)]:
                        tmp_remain = copy.deepcopy(one_state[2])
                        tmp_base = tmp_remain.pop(0)
                        for dec in transform(tmp_base, encode_list, func_idx):
                            if dec.shape != base_img_shape:
                                continue
                            all_middle_result.append([dec, 0, tmp_remain])
                for _ in range(row - 1):
                    tmp_all_middle_result = list()
                    for _, one_middle_result in enumerate(all_middle_result):
                        for idx, candidate_img in enumerate(one_middle_result[2]):
                            for decoded_remain_img in transform(candidate_img, encode_list, func_idx):
                                tmp_remain_images = copy.deepcopy(one_middle_result[2])
                                if decoded_remain_img.shape[1] != decoded_new_base_img.shape[1]:
                                    continue

                                pasted_img, pasted_state = paste_row_wise(one_middle_result[0], decoded_remain_img)
                                tmp_measure = pasted_state[0] + one_row_result[1]
                                tmp_remain_images.pop(idx)
                                tmp_all_middle_result.append([pasted_img, tmp_measure, tmp_remain_images])
                    all_middle_result = copy.deepcopy(tmp_all_middle_result)
                col_state[str(c)] = all_middle_result

            print("a")



    """
    new_all_row_result = list()
    best_measure = None
    best_image = None
    n = 0
    for one_row_result in all_row_result:
        tmp_remain_images = copy.deepcopy(one_row_result[2])
        new_base_img = tmp_remain_images.pop(0)

        for new_base_decode_idx, new_base_decode_img in enumerate(transform(new_base_img, encode_list, func_idx)):

            if base_img_shape != new_base_decode_img.shape:
                continue

            for remain_image in tmp_remain_images:
                for remain_img_idx, remain_decode_img in enumerate(transform(remain_image, encode_list, func_idx)):

                    if new_base_decode_img.shape[1] != remain_decode_img.shape[1]:
                        continue

                    pasted_img, pasted_state = paste_row_wise(new_base_decode_img, remain_decode_img)
                    tmp_measure = one_row_result[1] + pasted_state[0]

                    right_measure = 0
                    left_measure = 0
                    for i in range(pasted_img.shape[0]):
                        right_measure += euclidean_distance(one_row_result[0][i, -1, :], pasted_img[i, 0, :])
                        left_measure += euclidean_distance(one_row_result[0][i, 0, :], pasted_img[i, -1, :])
                    right_measure = (-1 * right_measure/pasted_img.shape[0]) + tmp_measure
                    left_measure = (-1 * left_measure/pasted_img.shape[0]) + tmp_measure

                    result_state = [right_measure, 'right'] if right_measure > left_measure else [left_measure, 'left']

                    stacked_img = None
                    if result_state[1] == 'right':
                        stacked_img = np.hstack((one_row_result[0], pasted_img))
                    elif result_state[1] == 'left':
                        stacked_img = np.hstack((pasted_img, one_row_result[0]))

                    # print(n, ":", one_row_result[1], pasted_state[0], result_state[0])
                    # cv2.imwrite("./%d.jpg" % n, stacked_img)
                    # n+=1
                    if not best_measure or result_state[0] > best_measure:
                        best_measure = result_state[0]
                        best_image = stacked_img

    cv2.imwrite("./final.jpg", best_image)
    """
