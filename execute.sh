#!/bin/bash
python cut_image.py --image_name './img/cat_img.jpg' --num_col 2 --num_row 2 --output_dir './'
python paste_image.py --input_dir './cut_image' --num_col 2 --num_row 2 --output_dir './'
