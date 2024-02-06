import skimage.measure
from skimage.filters import gaussian
import skimage.morphology
import numpy as np
from skimage.segmentation import watershed
import zarr
from scipy.ndimage import binary_fill_holes
from skimage.transform import resize
from skimage.morphology import dilation, cube
import skimage.util
from funlib.persistence import open_ds, prepare_ds
import daisy
from zarr.errors import PathNotFoundError

import scipy.ndimage

input_file = (
    "/nrs/cellmap/pattonw/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5"
)
output_file = (
    "/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1_postprocessed.n5"
)
dataset = "/predictions/2023-04-10/er/best"
out_dataset = "er/postprocessed_er"

context_padding = 10
threshold = 0.5
# min_size = 7e6 / (8**3)
gaussian_kernel = 2

cell_settings =  (
        "/nrs/cellmap/data/jrc_mus-liver-zon-1/staging/jrc_mus-liver-zon-1_prediction_edits.zarr",
        "s0",
        "background",
        cube(10),
    )
# =========
array_in = open_ds(input_file, dataset)
masks = []
cell_file, cell_dataset, cell_type, structure_element = cell_settings
try:
    cell_mask = open_ds(cell_file, cell_dataset)
except PathNotFoundError as e:
    print(f"Mask {cell_dataset} not found in {cell_file}")
    raise e
masks.append((cell_mask, cell_type, structure_element))
voxel_size = array_in.voxel_size
block_size = np.array(array_in.data.chunks) * np.array(voxel_size)
write_size = daisy.Coordinate(block_size)
context = daisy.Coordinate(np.array(voxel_size) * context_padding)

write_roi = daisy.Roi((0,) * len(write_size), write_size)
read_roi = write_roi.grow(context, context)
total_roi = array_in.roi.grow(context, context)
# ======== TEST ROI:
# total_roi = daisy.Roi((172800, 38400, 44800), voxel_size * 1024)
# total_roi = daisy.Roi(
#     voxel_size * daisy.Coordinate(17329, 9978, 7708), voxel_size * 2048
# )
# ===========
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
    instance = array_in.to_ndarray(block.read_roi, fill_value=0).astype(np.int64)

    # threshold
    instance = (instance > threshold).astype(np.int64)

    instance = gaussian(instance, sigma=gaussian_kernel)

    # make mask of unwanted object class overlaps
    all_mask = np.zeros(instance.shape, dtype=np.uint8)
    for mask, mask_type, structure_element in masks:
        # get mask for this block expanding to fit mask voxel size
        this_read_roi = block.read_roi.snap_to_grid(mask.voxel_size, mode="grow")
        if mask_type == "foreground":
            this_mask = mask.to_ndarray(this_read_roi, fill_value=0) > 0
        elif mask_type == "background":
            this_mask = mask.to_ndarray(this_read_roi, fill_value=0) == 0
        else:
            raise ValueError(f"Unknown mask type {mask_type}")
        # RESIZE MASK TO MATCH INSTANCE VOXEL SIZE THEN CROP TO MATCH BLOCK
        resampled_shape = this_read_roi.shape / voxel_size
        this_mask = (
            resize(
                this_mask,
                resampled_shape,
                order=0,
                anti_aliasing=False,
                mode="constant",
                cval=0,
                clip=True,
                preserve_range=True,
            )
            > 0
        )
        crop_width = (
            (np.array(this_mask.shape) - np.array(instance.shape)) // 2
        ).astype(int)
        crop_width = [[cw, cw] for cw in crop_width]
        this_mask = skimage.util.crop(this_mask, crop_width)
        this_mask = dilation(this_mask, structure_element)
        all_mask[this_mask] = 1
    print(f"done making mask, shape: {all_mask.shape}")

    # find ids of peroxisomes that overlap with unwanted object classes
    # bad_ids = np.unique(instance[all_mask])
    # print("bad ids")
    # set bad ids to 0
    # for id in bad_ids:
    #     instance[instance == id] = 0
    # faster
    # bad_id_mask = np.isin(instance, bad_ids)
    # print("bad id mask")

    # Set all bad ids to 0
    # instance[all_mask] = 0
    instance[all_mask == 1] = 0
    print("done masking")

    #  set cell id to the instance
    # labeled_cells, num_cells = scipy.ndimage.label(cell_mask)
    instance[instance>0] = cell_mask[instance>0]
    print("done cell ids")

    # remove small objects << CRASHES HERE ?? >>
    # instance = skimage.morphology.remove_small_objects(
    #     instance, min_size=int(np.ceil(min_size))
    # )
    # for id in np.unique(instance):
    #     if np.sum(instance == id) < min_size:
    #         instance[instance == id] = 0
    print("done remove small")
    # relabel objects to smallest range of integers
    # instance = skimage.measure.label(instance)
    print("done relabel")
    # print("done relabel", instance.shape)

    return instance.astype(np.uint64)
