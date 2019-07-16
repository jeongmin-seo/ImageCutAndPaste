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
parser.add_argument('--measure', type=str, default='euclidean')
parser.add_argument('--output_dir', metavar='PATH', type=str, default='./')


def edge_similar(_img, _coordinate, _direction='row'):
    """
    :param _img: Input RGB image.
    :param _coordinate: Coordinate related to the position at which to count the number of edge pixels.
    :param _direction: The factor related of the direction of counting the number of edge pixels.
    :return: Number of pixels that can be judged to be edges at the image boundary.
    """
    n = 0
    _copied_img = copy.deepcopy(_img)
    _copied_img = cv2.cvtColor(_copied_img, cv2.COLOR_RGB2GRAY)
    if _direction == 'row':
        edge = cv2.Laplacian(_copied_img, cv2.CV_8U, ksize=5)
        for i in range(-1, 2):
            for detected_edge in edge[_coordinate[0] + i, :]:
                if detected_edge > 250 or detected_edge < 6:
                    n += 1

    else:
        edge = cv2.Laplacian(_copied_img, cv2.CV_8U, ksize=5)
        for i in range(-1, 2):
            for detected_edge in edge[:, _coordinate[1] + i]:
                if detected_edge > 250 or detected_edge < 6:
                    n += 1
    return (-1) * n


def euclidean_distance(_vector1, _vector2):
    """
    :param _vector1:  Input vector
    :param _vector2:  Input vector
    :return:  Squared value of euclidean distance.
    """
    assert _vector1.shape == _vector2.shape
    return np.sum((_vector1 - _vector2) ** 2)


def cos_similarity(_vector1, _vector2):
    """
    :param _vector1:  Input vector
    :param _vector2:  Input vector
    :return:  Cosine similarity value.
    """
    assert _vector1.shape == _vector2.shape
    return np.dot(_vector1, _vector2) / (np.linalg.norm(_vector1) * np.linalg.norm(_vector2))


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


def calc_measure(_factor, _measure='cosine'):
    """
    :param _factor:  Two vector lists to calculate.
    :param _measure: Factor related to the method of calculating similarity.
    :return: Calculated similarity.
    """
    measure = 0
    if _measure == 'cosine':
        for i in range(_factor[0].shape[-1]):
            if not i:
                measure = cos_similarity(_factor[0][i], _factor[1][i])
            else:
                measure += cos_similarity(_factor[0][i], _factor[1][i])
    elif _measure == 'euclidean':
        for i in range(_factor[0].shape[-1]):
            if not i:
                measure = euclidean_distance(_factor[0][i], _factor[1][i])
            else:
                measure += euclidean_distance(_factor[0][i], _factor[1][i])
        measure = (-1) * np.sqrt(measure)

    return measure


def decode_image(_image):
    """
    :param _image: Input RGB or Gray image.
    :return: Decoded (mirroring, flipping, rotate) image.
    """
    for _idx in func_idx:
        decoded_image = copy.deepcopy(_image)
        for i in range(len(_idx) - 1, -1, -1):
            decoded_image = decode_list[i](decoded_image)
        yield decoded_image


def col_wise_paste(_base_image, _compare_images, _col, _sim_measure):
    """
    :param _base_image: An image that is the basis for pasting an image in the row direction.
    :param _compare_images: Image list to compare with _base_image input by _base_image.
    :param _col: Number of columns.
    :param _sim_measure: A factor that determines which similarity calculation to take.
    :return: An image that connected in the column direction according to the similarity. (Final Result)
    """
    for _ in range(_col - 1):
        best_status = dict()
        for _compare_idx, _compare_image in enumerate(_compare_images):
            for tmp_image in decode_image(_compare_image):

                # Performs operations only when the size of the attached faces is the same.
                if _base_image.shape[0] == tmp_image.shape[0]:
                    right_measure = None
                    left_measure = None
                    # Determine whether to attach to the left or right side of the base image
                    # and get the best similarity.
                    if _sim_measure in ['euclidean', 'cosine']:
                        right_paste_factor = [_base_image[:, -1, :], tmp_image[:, 0, :]]
                        left_paste_factor = [_base_image[:, 0, :], tmp_image[:, -1, :]]

                        right_measure = calc_measure(right_paste_factor, 'euclidean')
                        left_measure = calc_measure(left_paste_factor, 'euclidean')

                    elif _sim_measure == 'edge':
                        # for edge similar
                        right_paste_factor = np.hstack((_base_image, tmp_image))
                        left_paste_factor = np.hstack((tmp_image, _base_image))

                        # TODO:
                        right_measure = edge_similar(right_paste_factor, _base_image.shape, 'col')
                        left_measure = edge_similar(left_paste_factor, tmp_image.shape, 'col')
                    else:
                        raise ValueError

                    paste_status = ["right", right_measure] if right_measure > left_measure else ["left", left_measure]

                    if not best_status or paste_status[1] > best_status['measure']:
                        best_status['image'] = tmp_image
                        best_status['measure'] = paste_status[1]
                        best_status['loc'] = paste_status[0]
                        best_status['idx'] = _compare_idx

        # paste to the base image according to the determined image and orientation.
        if best_status['loc'] == 'right':
            _base_image = np.hstack((_base_image, best_status['image']))
        else:
            _base_image = np.hstack((best_status['image'], _base_image))

        # exclude pasted images from candidate images.
        _compare_images.pop(best_status['idx'])

    return _base_image


def row_wise_paste(_base_image, _compare_images, _row, _sim_measure):
    """
    :param _base_image: An image that is the basis for pasting an image in the row direction.
    :param _compare_images: Image list to compare with _base_image input by _base_image.
    :param _row: Number of rows.
    :param _sim_measure: A factor that determines which similarity calculation to take.
    :return: 1. An image that connected in the row direction according to the similarity.
             2. Calculated maximum similarity.
             3. Remaining compare images.
    """
    best_measure = None
    for _ in range(_row - 1):
        best_status = dict()
        for _compare_idx, _compare_image in enumerate(_compare_images):
            for tmp_image in decode_image(_compare_image):

                # Determine whether to attach to the top or bottom side of the base image
                # and get the best similarity
                if _base_image.shape[1] == tmp_image.shape[1]:
                    top_measure = None
                    bottom_measure = None
                    if _sim_measure in ['euclidean', 'cosine']:
                        top_paste_factor = [_base_image[0, :, :], tmp_image[-1, :, :]]
                        bottom_paste_factor = [_base_image[-1, :, :], tmp_image[0, :, :]]

                        top_measure = calc_measure(top_paste_factor, 'euclidean')
                        bottom_measure = calc_measure(bottom_paste_factor, 'euclidean')

                    elif _sim_measure == 'edge':
                        # for edge similar
                        top_paste_factor = np.vstack((tmp_image, _base_image))
                        bottom_paste_factor = np.vstack((_base_image, tmp_image))

                        top_measure = edge_similar(top_paste_factor, tmp_image.shape, 'row')
                        bottom_measure = edge_similar(bottom_paste_factor, _base_image.shape, 'row')

                    paste_status = ["top", top_measure] if top_measure > bottom_measure else ["bottom", bottom_measure]

                    if not best_status or paste_status[1] > best_status['measure']:
                        best_status['image'] = tmp_image
                        best_status['measure'] = paste_status[1]
                        best_status['loc'] = paste_status[0]
                        best_status['idx'] = _compare_idx

        # paste to the base image according to the determined image and orientation.
        if best_status['loc'] == 'top':
            _base_image = np.vstack((best_status['image'], _base_image))
        else:
            _base_image = np.vstack((_base_image, best_status['image']))

        # exclude pasted images from candidate images.
        _compare_images.pop(best_status['idx'])
        best_measure = best_status['measure']
    return _base_image, best_measure, _compare_images


def mirroring(_img):
    return cv2.flip(_img, 1)


def flipping(_img):
    return cv2.flip(_img, 0)


def unrotate(_img):
    import cv2
    return cv2.rotate(_img, cv2.ROTATE_90_CLOCKWISE)


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


if __name__ == "__main__":

    global args
    args = parser.parse_args()

    input_dir = args.input_dir
    row = args.num_row
    col = args.num_col
    selected_measure = args.measure
    output_name = os.path.join(args.output_dir, 'reconstructed_image.jpg')

    decode_list = [mirroring, flipping, unrotate]
    func_idx = index_combination(range(len(decode_list)))

    image_list = load_cut_images(input_dir)
    base_image = image_list.pop(0)
    base_shape = copy.deepcopy(base_image.shape)

    if (base_shape[0] < base_shape[1] and col > row) or \
            (base_shape[0] > base_shape[1] and col < row):
        tmp = copy.deepcopy(row)
        row = copy.deepcopy(col)
        col = tmp

    row_iamges = list()
    for i in range(col):
        row_image = None
        if not i:
            base_image, _, image_list = row_wise_paste(base_image, image_list, row, selected_measure)
            row_image = base_image

        else:
            best_similarity = None
            base_image = image_list.pop(0)
            tmp_image_list = list()
            for idx in func_idx:
                base_tmp_image = copy.deepcopy(base_image)
                # base_tmp_image = copy.deepcopy(row_image)
                # decode images using decode function
                for j in idx:
                    base_tmp_image = decode_list[j](base_tmp_image)

                copied_image_list = copy.deepcopy(image_list)
                if base_tmp_image.shape == base_shape:
                    result_image, similarity, copied_image_list = \
                        row_wise_paste(base_tmp_image, copied_image_list, row, selected_measure)

                    if not best_similarity or best_similarity < similarity:
                        best_similarity = similarity
                        row_image = result_image
                        tmp_image_list = copied_image_list

            image_list = tmp_image_list
        row_iamges.append(row_image)

    base_image = row_iamges.pop(0)
    return_image = col_wise_paste(base_image, row_iamges, col, selected_measure)
    cv2.imwrite(output_name, return_image)
