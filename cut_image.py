import cv2
import os
import random
import string
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_name', metavar='PATH', type=str, default='./cut_img.jpg')
parser.add_argument('--num_col', type=int, default=3)
parser.add_argument('--num_row', type=int, default=3)
parser.add_argument('--output_dir', metavar='PATH', type=str, default='./')


def encode_image(_img):
    enc_names = ['mirroring', 'flipping', 'rotation']
    for enc_name in enc_names:
        if random.randint(0, 1):
            if enc_name == 'mirroring':
                _img = cv2.flip(_img, 1)
            elif enc_name == 'flipping':
                _img = cv2.flip(_img, 0)
            else:
                _img = cv2.rotate(_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return _img


def cut_image(_img, _row, _col):
    _cut_imgs = list()
    (height, width, _) = _img.shape
    cut_height = height//_row
    cut_width = width//_col
    for r_idx in range(_row):
        for c_idx in range(_col):
            cut_img = _img[r_idx*cut_height: (r_idx +1)*cut_height, c_idx*cut_width: (c_idx+1)*cut_width, :]
            _cut_imgs.append(cut_img)

    return _cut_imgs


def generate_name(_length=8):
    result = ""
    for _ in range(_length):
        result += random.choice(string.ascii_lowercase + string.digits)
    return result + ".jpg"


if __name__ == "__main__":
    global args
    args = parser.parse_args()

    input_path = args.image_name
    output_path = os.path.join(args.output_dir, 'cut_image')
    row = args.num_row
    col = args.num_col

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    img = cv2.imread(input_path)
    cut_imgs = cut_image(img, row, col)

    for i, cut_img in enumerate(cut_imgs):
        encoded_img = encode_image(cut_img)
        random_name = generate_name()
        print(random_name)
        cv2.imwrite(os.path.join(output_path, random_name), encoded_img)

