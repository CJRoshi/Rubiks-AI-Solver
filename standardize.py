import numpy as np
import cv2
import os

file_folder = 'dataset_imgs'

for filename in os.listdir(file_folder):
    true_filename = file_folder+"/"+filename
    img = cv2.imread(true_filename)

    imgheight, imgwidth, depth = np.shape(img)

    if imgwidth >= imgheight:
        larger_side = imgwidth
        smaller_side = imgheight

        border_top  = (smaller_side)//8
        border_side = (larger_side)//2-3*border_top
        true_center_width = 4*border_top
        center_width = 6*border_top

    elif imgwidth < imgheight:
        larger_side = imgheight
        smaller_side = imgwidth

        border_side  = (smaller_side)//8
        border_top = (larger_side)//2-3*border_side
        true_center_width = 4*border_side
        center_width = 6*border_side

    # Crop the img to the above dimensions.
    img_cropped = np.copy(img)
    img_cropped = img_cropped[border_top:border_top+center_width, border_side:border_side+center_width]

    cv2.imwrite(true_filename, img_cropped)