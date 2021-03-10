import os
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm
from skimage.io import imsave

from src.ml.model import HUNet
from src.ml.train import restore_latest_model_epoch, normalize_batch
from src.ml.loss import dice, iou
from src.config import PATCH_SHAPE, DATA_DIR
from src.image import read_tiff


def segment_volume_at_coordinates(model,
                                  compute_device,
                                  image_volume,
                                  result_volume,
                                  grid_coordinates,
                                  overlap_z,
                                  overlap_xy):
    # iterate through grid_coordinates in batches of reasonable size
    # the significance of the batches is they are what will go to the model for prediction at one time
    batch_size = 12
    for batch_start in tqdm(range(0, len(grid_coordinates), batch_size)):
        batch_corner_coords = grid_coordinates[batch_start:batch_start + batch_size]
        # NOTE: the final batch will be less than batch_size unless exactly divisible by len(grid_coordinates)
        true_batch_size = len(batch_corner_coords)

        # create a batch of image patches from coordinates
        image_patches = np.zeros((true_batch_size, PATCH_SHAPE[0], PATCH_SHAPE[1], PATCH_SHAPE[2]))
        for i, corner_coord in enumerate(batch_corner_coords):
            image_patch = image_volume[
                          corner_coord[0]:corner_coord[0] + PATCH_SHAPE[0],
                          corner_coord[1]:corner_coord[1] + PATCH_SHAPE[1],
                          corner_coord[2]:corner_coord[2] + PATCH_SHAPE[2],
                          ]
            image_patches[i, :, :, :] = image_patch

        # normalize and then predict on batch
        image_patches = normalize_batch(image_patches)
        image_patches = torch.from_numpy(image_patches.astype(np.float32)).to(compute_device)
        with torch.no_grad():
            predictions = model(image_patches).cpu().detach().numpy()

        # iterate through patches in batch and place them in the result image volume
        for i, corner_coord in enumerate(batch_corner_coords):
            # crop out the patch overlap (remove the perimeter)
            cropped_image_patch = predictions[i,
                                              overlap_z:-overlap_z,
                                              overlap_xy:-overlap_xy,
                                              overlap_xy:-overlap_xy]
            # insert that crop into the right place in the result image
            result_volume[
                corner_coord[0] + overlap_z:corner_coord[0] + PATCH_SHAPE[0] - overlap_z,
                corner_coord[1] + overlap_xy:corner_coord[1] + PATCH_SHAPE[1] - overlap_xy,
                corner_coord[2] + overlap_xy:corner_coord[2] + PATCH_SHAPE[2] - overlap_xy
            ] = cropped_image_patch


def segment_stack(model, image_stack):
    """
    The idea here is to pad the entire stack on both sides, such that sliding a window
    across and not quite covering the volume (for example because the stack isn't neatly
    divisible in to patch sized chunks) will still cover the initial volume.

    There is overlap in the window positions, and some of the overlap is discarded from
    each side.
    """
    if torch.cuda.is_available():
        compute_device = torch.device('cuda')
        model.cuda()
    else:
        compute_device = torch.device('cpu')

    overlap_xy = PATCH_SHAPE[1] // 4
    overlap_z = PATCH_SHAPE[0] // 4

    # padding guarantees that the whole volume is convolved over.
    padded_volume = np.pad(
        image_stack,
        (
            (overlap_z+PATCH_SHAPE[0], overlap_z+PATCH_SHAPE[0]),
            (overlap_xy+PATCH_SHAPE[1], overlap_xy+PATCH_SHAPE[1]),
            (overlap_xy+PATCH_SHAPE[2], overlap_xy+PATCH_SHAPE[2])
        ),
        'symmetric'
    )

    # This creates a list of corner coordinates for image patches to start from
    # it starts from the left/upmost edge of the padded image stack, and iterates
    # through in steps of (patch shape - overlap). It stops a full patch shape
    # before the end of the padded volume so that the model window always falls
    # inside the volume.
    grid_coordinates = [
        (z, y, x)
        for z in range(0, padded_volume.shape[0]-PATCH_SHAPE[0], PATCH_SHAPE[0]-(overlap_z*2))
        for y in range(0, padded_volume.shape[1]-PATCH_SHAPE[1], PATCH_SHAPE[1]-(overlap_xy*2))
        for x in range(0, padded_volume.shape[2]-PATCH_SHAPE[2], PATCH_SHAPE[2]-(overlap_xy*2))
    ]
    result_volume = np.zeros_like(padded_volume, dtype=np.float16)

    segment_volume_at_coordinates(model,
                                  compute_device,
                                  padded_volume,
                                  result_volume,
                                  grid_coordinates,
                                  overlap_z,
                                  overlap_xy)

    # remove the padding to restore original shape
    result_volume = result_volume[
        overlap_z+PATCH_SHAPE[0]:-(overlap_z+PATCH_SHAPE[0]),
        overlap_xy+PATCH_SHAPE[1]:-(overlap_xy+PATCH_SHAPE[1]),
        overlap_xy+PATCH_SHAPE[2]:-(overlap_xy+PATCH_SHAPE[2]),
    ]

    return result_volume


def run_metrics(stack1: torch.Tensor, stack2: torch.Tensor) -> Dict[str, float]:
    return {
        'dice': float(dice(stack1, stack2, smooth=0)),
        'iou': float(iou(stack1, stack2)),
    }


def segment_save_and_print_metrics(model,
                                   stack_name: str,
                                   save_folder: str,
                                   image_stack: np.ndarray,
                                   benchmark_segmentations: np.ndarray,
                                   save_postfix=""):
    segmentation_stack = segment_stack(model, image_stack)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    imsave(os.path.join(save_folder, stack_name+save_postfix+'.tiff'), segmentation_stack)

    metrics = {}
    if benchmark_segmentations is not None:
        # convert to pytorch objects since metrics accept that datatype
        benchmark_segmentations = torch.from_numpy(benchmark_segmentations.astype(np.float32))
        segmentation_stack = torch.from_numpy(segmentation_stack.astype(np.float32))
        # convert to binary labels
        benchmark_segmentations = (benchmark_segmentations > 0.5).long()
        segmentation_stack = (segmentation_stack > 0.5).long()

        metrics = run_metrics(segmentation_stack, benchmark_segmentations)
        print(f"Results for running metrics on {stack_name}...")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
    return metrics


