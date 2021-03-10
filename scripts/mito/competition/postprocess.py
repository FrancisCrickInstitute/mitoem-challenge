import os
import argparse
import numpy as np
from skimage.io import imread, imsave
from skimage.measure import label
import h5py


#gt_fp = "mito_r_val_gt.tiff"
#gt = imread(gt_fp)
#print(f"gt max: {np.max(gt)}")
#print(f"gt n: {len(np.unique(gt))}")
#del gt

#boundary_fp = "../Everything_summed_boundary.tif"
#area_fp = "../Everything_summed_area.tif"

#ws_save_name = "ws_everything_summed"

parser = argparse.ArgumentParser(description='')
parser.add_argument('--area', type=str, required=True)
parser.add_argument('--boundary', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()
area_fp = args.area
boundary_fp = args.boundary
ws_save_name = args.output

###############
### CC
###############

print("\nreading boundary")
boundary = imread(boundary_fp)
print("\nreading area")
area = imread(area_fp)

area_bin = area > 0.5
boundary_bin = boundary > 0.5

area_minus_boundary = np.logical_and(area_bin == 1, boundary_bin == 0).astype(np.uint16)

print('doing cc')
amb_cc = label(area_minus_boundary).astype(np.uint16)
#amb_cc_save_fp = "amb_cc_mito_r_val.tiff"
#imsave(amb_cc_save_fp, amb_cc)

print(f"amb cc max: {np.max(amb_cc)}")
print(f"amb cc n: {len(np.unique(amb_cc))}")

##################
### WATERSHED
##################

from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

print('doing watershed')
ws_w_small = watershed(image=boundary, markers=amb_cc, mask=area_bin.astype(np.uint16))

print('removing small')

ws_removed = remove_small_objects(ws_w_small, min_size=1500)
ws_removed = ws_removed.astype(np.uint16)


print('watershed done, saving')
ws_save_fp = ws_save_name+".tiff"
imsave(ws_save_fp, ws_removed, check_contrast=False, compress=True)

print('finished')

########################
###  CONVERT TO H5
########################

# Create the h5 file (using lzf compression to save space)
h5f = h5py.File(ws_save_name+'.h5', 'w')
h5f.create_dataset('dataset_1', data=ws_removed, compression="lzf")
h5f.close()





