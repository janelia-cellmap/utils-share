import skimage.measure
import skimage.filters
import skimage.morphology
from skimage.segmentation import watershed
from skimage.morphology import dilation, cube, ball
from scipy import ndimage as ndi
import daisy
import numpy as np
from funlib.persistence import open_ds, prepare_ds

input_file = "/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1_postprocessed.n5"
output_file = (
    "/nrs/cellmap/rhoadesj/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5"
)
dataset = "mito/mito"
out_dataset = "mito/mito_global"
array_in = open_ds(input_file, dataset)
voxel_size = array_in.voxel_size
block_size = np.array(array_in.data.chunks) * np.array(voxel_size)
write_size = daisy.Coordinate(block_size)
context = daisy.Coordinate(np.array(voxel_size) * 10)  # 50 pixel overlap
threshold = 0.58
# min_size = 2e7 / (8**3)
gaussian_kernel = 2

write_roi = daisy.Roi((0,) * len(write_size), write_size)
read_roi = write_roi.grow(context, context)
total_roi = array_in.roi.grow(context, context)

num_voxels_in_block = (read_roi / array_in.voxel_size).size

# try:
#     array_out = open_ds(output_file, out_dataset, mode="a")
# except KeyError:
#     array_out = prepare_ds(
#         output_file,
#         out_dataset,
#         total_roi,
#         voxel_size=voxel_size,
#         write_size=write_size,
#         dtype=np.uint64,
#     )


def segment_function(block):
    instance = array_in.to_ndarray(block.read_roi, fill_value=0)
    return instance.astype(np.uint64)
