import logging
import subprocess
import time


import skimage.measure
import skimage.filters
import skimage.morphology
from skimage.segmentation import watershed
from skimage.morphology import dilation, cube, ball
from scipy import ndimage as ndi
import daisy
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from funlib.persistence import open_ds

num_workers = 64
input_file = (
    "/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5"
)
# output_file = '/groups/cosem/cosem/ackermand/Cryo_FS80_Cell2_4x4x4nm_setup03_it1100000.n5'
output_file = "/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1_postprocessed.n5"
dataset = "mito/mito"
out_dataset = "mito/postprocessed_mito"
array_in = open_ds(input_file, dataset)
array_out = open_ds(output_file, out_dataset, mode="a")
voxel_size = array_in.voxel_size
context = np.array(voxel_size) * 50  # 50 pixel overlap
block_size = np.array(array_in.data.chunks) * np.array(voxel_size)
threshold = 0.58
# min_size = 2e7 / (8**3)
gaussian_kernel = 2

write_size = daisy.Coordinate(block_size)
write_roi = daisy.Roi((0,) * len(write_size), write_size)
read_roi = write_roi.grow(context, context)
total_roi = array_in.roi.grow(context, context)

num_voxels_in_block = (read_roi / array_in.voxel_size).size


def segment_function(block):
    mito_dist = array_in.to_ndarray(block.read_roi, fill_value=0)
    mito_dist = skimage.filters.gaussian(mito_dist, sigma=gaussian_kernel)
    logger.info("done gaussian", mito_dist.shape, mito_dist.max(), mito_dist.min())
    binary_peroxi = mito_dist > threshold
    logger.info("done threshold", binary_peroxi.shape)
    markers, _ = ndi.label(binary_peroxi)
    # Apply Watershed
    ws_labels = watershed(-mito_dist, markers, mask=binary_peroxi)
    logger.info("done watershed", ws_labels.shape)
    instance_peroxi = skimage.measure.label(ws_labels).astype(np.int64)
    logger.info(
        "done instance",
        instance_peroxi.shape,
        instance_peroxi.max(),
        instance_peroxi.min(),
    )
    instance_peroxi[binary_peroxi == 0] = 0
    logger.info("done mask instance", instance_peroxi.shape)
    instance_peroxi = dilation(instance_peroxi, cube(2))
    #  dilation(LD[:], cube(2))

    return instance_peroxi.astype(np.uint64)
