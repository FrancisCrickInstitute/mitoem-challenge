import os

import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
import cv2 as cv


# human
RAW_IMAGE_FOLDER = "MitoEM-H/im/"
TRAIN_LABEL_IMAGE_FOLDER = "MitoEM-H/mito_train/"
VAL_LABEL_IMAGE_FOLDER = "MitoEM-H/mito_val/"
SAVE_FNAME_PREFIX = "mito_h_"

# rat
# RAW_IMAGE_FOLDER = "MitoEM-R/im/"
# TRAIN_LABEL_IMAGE_FOLDER = "MitoEM-R/mito_train/"
# VAL_LABEL_IMAGE_FOLDER = "MitoEM-R/mito_val/"
# SAVE_FNAME_PREFIX = "mito_r_"


def z_pad(z):
    z = str(z)
    if len(z) == 1:
        return '000' + z
    elif len(z) == 2:
        return '00' + z
    elif len(z) == 3:
        return '0' + z
    return z


def split_and_save_stack(save_fname_prefix, save_fname_postfix, combined_stack):
    save_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'stacks'))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # save 4 of 100x2048x2048
    print(combined_stack.dtype)
    print(f"min: {np.min(combined_stack)}")
    print(f"max: {np.max(combined_stack)}")
    for x in (0, 2048):
        for y in (0, 2048):
            save_fname = SAVE_FNAME_PREFIX+save_fname_prefix + f'_x{z_pad(x)}_y_{z_pad(y)}'
            if save_fname_postfix != '':
                save_fname += f'_{save_fname_postfix}'
            save_fname += '.tiff'
            save_fname = os.path.join(save_folder, save_fname)
            substack = combined_stack[:, x:x+2048, y:y+2048]
            print(f'saving {save_fname}')
            imsave(save_fname, substack)


def get_boundary_image_from_label(label_im):
    out_boundary_im = np.zeros(label_im.shape, dtype=np.uint16)
    for label_idx in np.unique(label_im):
        # 0 is the background
        if label_idx != 0:
            shape_at_idx = (label_im == label_idx).astype(np.uint8)
            contours, _ = cv.findContours(shape_at_idx, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cv.drawContours(out_boundary_im, contours, -1, 1, 5)
    return out_boundary_im


def stack_for_area_and_boundary(fname_prefix, folder, ext, dtype, start, end):
    print(f"Loading {fname_prefix} between {start} and {end} from {folder}")
    out_area = np.zeros((100, 4096, 4096), dtype=dtype)
    out_boundary = np.zeros((100, 4096, 4096), dtype=dtype)
    for i in tqdm(range(start, end)):
        fname = fname_prefix+z_pad(i)+ext
        area_label_im = imread(os.path.join(folder, fname))
        boundary_im = get_boundary_image_from_label(area_label_im)
        # make area label image binary... but not before using labels to get boundaries
        area_label_im = (area_label_im > 0).astype(np.uint8)
        # put area and boundary in output stacks
        out_area[i - start, :, :] = area_label_im
        out_boundary[i-start, :, :] = boundary_im
    print(f"out area dtype: {out_area.dtype}")
    print(f"out boundary dtype: {out_boundary.dtype}")
    return out_area, out_boundary


def stack_images_in_range(fname_prefix, folder, ext, dtype, start, end):
    print(f"Loading {fname_prefix} between {start} and {end} from {folder}")
    out_stack = np.zeros((100, 4096, 4096), dtype=dtype)
    for i in tqdm(range(start, end)):
        fname = fname_prefix+z_pad(i)+ext
        im = imread(os.path.join(folder, fname))
        out_stack[i-start, :, :] = im
    print(out_stack.dtype)
    return out_stack


def main():
    # training data
    stack = stack_images_in_range('im', RAW_IMAGE_FOLDER, '.png', np.uint8, 0, 100)
    split_and_save_stack('train_z000', '', stack)
    area, boundary = stack_for_area_and_boundary('seg', TRAIN_LABEL_IMAGE_FOLDER, '.tif', np.uint16, 0, 100)
    split_and_save_stack('train_z000', 'area', area)
    split_and_save_stack('train_z000', 'boundary', boundary)

    stack = stack_images_in_range('im', RAW_IMAGE_FOLDER, '.png', np.uint8, 100, 200)
    split_and_save_stack('train_z100', '', stack)
    area, boundary = stack_for_area_and_boundary('seg', TRAIN_LABEL_IMAGE_FOLDER, '.tif', np.uint16, 100, 200)
    split_and_save_stack('train_z100', 'area', area)
    split_and_save_stack('train_z100', 'boundary', boundary)

    stack = stack_images_in_range('im', RAW_IMAGE_FOLDER, '.png', np.uint8, 200, 300)
    split_and_save_stack('train_z200', '', stack)
    area, boundary = stack_for_area_and_boundary('seg', TRAIN_LABEL_IMAGE_FOLDER, '.tif', np.uint16, 200, 300)
    split_and_save_stack('train_z200', 'area', area)
    split_and_save_stack('train_z200', 'boundary', boundary)

    stack = stack_images_in_range('im', RAW_IMAGE_FOLDER, '.png', np.uint8, 300, 400)
    split_and_save_stack('train_z300', '', stack)
    area, boundary = stack_for_area_and_boundary('seg', TRAIN_LABEL_IMAGE_FOLDER, '.tif', np.uint16, 300, 400)
    split_and_save_stack('train_z300', 'area', area)
    split_and_save_stack('train_z300', 'boundary', boundary)

    # validation data
    stack = stack_images_in_range('im', RAW_IMAGE_FOLDER, '.png', np.uint8, 400, 500)
    split_and_save_stack('val', '', stack)
    area, boundary = stack_for_area_and_boundary('seg', VAL_LABEL_IMAGE_FOLDER, '.tif', np.uint16, 400, 500)
    split_and_save_stack('val', 'area', area)
    split_and_save_stack('val', 'boundary', boundary)


if __name__ == '__main__':
    main()


