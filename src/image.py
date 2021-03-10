import os

import numpy as np
from skimage.io import imread


def read_tiff(folder: str, tiff_filename: str) -> np.ndarray:
    tiff_loc = os.path.join(folder, tiff_filename+'.tiff')
    if os.path.isfile(tiff_loc):
        return imread(tiff_loc)
    tif_loc = os.path.join(folder, tiff_filename+'.tif')
    if os.path.isfile(tif_loc):
        return imread(tif_loc)

    raise FileNotFoundError(f"Couldn't find .tif or .tiff file {tiff_filename} in {folder}")

