#!/bin/bash

for row in 2 3 4
do
    for col in 2 3 4
    do
        mkdir "${row}_${col}"
        python cut_image.py --image_name './img/cat_img.jpg' --num_col ${col} --num_row ${row} --output_dir "./${row}_${col}"
        python paste_image.py --input_dir "./${row}_${col}/cut_image" --num_col ${col} --num_row ${row} --output_dir "./${row}_${col}"
    done
done
