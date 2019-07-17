import cv2
import os
import argparse
import copy
import itertools
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', metavar='PATH', type=str, default='./cut_image')
parser.add_argument('--num_col', type=int, default=2)
parser.add_argument('--num_row', type=int, default=2)
parser.add_argument('--output_dir', metavar='PATH', type=str, default='./')


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
    :param _decode_list: transform function list.
    :param _func_idx: transform function index combination.
    :return: Decoded (mirroring, flipping, rotate) image.
    """
    for _idx in _func_idx:
        decoded_image = copy.deepcopy(_image)
        for _i in range(len(_idx) - 1, -1, -1):
            decoded_image = _decode_list[_idx[_i]](decoded_image)
        yield decoded_image


if __name__ == "__main__":

    global args
    args = parser.parse_args()

    input_dir = args.input_dir
    row = args.num_row
    col = args.num_col
    output_dir = args.output_dir

    # dir_path = "./cut_image"
    loaded_images = load_cut_images(input_dir)

    decode_list = [mirroring, flipping, unrotate, rotate]
    func_idx = index_combination(range(len(decode_list)))

    base_img = loaded_images.pop(0)
    base_img_shape = base_img.shape
    all_row_result = list()

    for idx, candidate_img in enumerate(loaded_images):
        for used_img_idx, decode_img in enumerate(transform(candidate_img, decode_list, func_idx)):
            tmp_remain_images = copy.deepcopy(loaded_images)

            if decode_img.shape[1] != base_img.shape[1]:
                continue
            pasted_img, pasted_state = paste_row_wise(base_img, decode_img)
            tmp_remain_images.pop(idx)
            all_row_result.append([pasted_img, pasted_state[0], tmp_remain_images])

    new_all_row_result = list()
    best_measure = None
    best_image = None
    for one_row_result in all_row_result:
        tmp_remain_images = copy.deepcopy(one_row_result[2])
        new_base_img = tmp_remain_images.pop(0)

        for new_base_decode_idx, new_base_decode_img in enumerate(transform(new_base_img, decode_list, func_idx)):

            if base_img_shape != new_base_decode_img.shape:
                continue

            for remain_image in tmp_remain_images:
                for remain_img_idx, remain_decode_img in enumerate(transform(remain_image, decode_list, func_idx)):

                    if new_base_decode_img.shape[1] != remain_decode_img.shape[1]:
                        continue

                    pasted_img, pasted_state = paste_row_wise(new_base_decode_img, remain_decode_img)
                    tmp_measure = one_row_result[1] + pasted_state[0]

                    right_measure = 0
                    left_measure = 0
                    for i in range(pasted_img.shape[0]):
                        right_measure += euclidean_distance(one_row_result[0][i, -1, :], pasted_img[i, 0, :])
                        left_measure += euclidean_distance(one_row_result[0][i, 0, :], pasted_img[i, -1, :])
                    # print((-1 * right_measure/pasted_img.shape[0]), (-1 * left_measure/pasted_img.shape[0]))
                    right_measure = (-1 * right_measure/pasted_img.shape[0]) + tmp_measure
                    left_measure = (-1 * left_measure/pasted_img.shape[0]) + tmp_measure

                    result_state = [right_measure, 'right'] if right_measure > left_measure else [left_measure, 'left']

                    stacked_img = None
                    if result_state[1] == 'right':
                        stacked_img = np.hstack((one_row_result[0], pasted_img))
                    elif result_state[1] == 'left':
                        stacked_img = np.hstack((pasted_img, one_row_result[0]))

                    if not best_measure or result_state[0] > best_measure:
                        best_measure = result_state[0]
                        best_image = stacked_img

    cv2.imwrite(os.path.join(output_dir, 'reconstruct.jpg'), best_image)
