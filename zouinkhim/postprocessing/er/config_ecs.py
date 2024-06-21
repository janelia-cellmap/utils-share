from scipy import ndimage
import skimage.measure
from skimage.filters import gaussian
import skimage.morphology
import skimage.util
from skimage.transform import resize
import numpy as np
from skimage.segmentation import watershed
import zarr
from skimage.morphology import erosion, cube, remove_small_holes, ball
from funlib.persistence import open_ds, prepare_ds
import daisy

import logging

# Set the logging level of the root logger to DEBUG
logging.basicConfig(level=logging.DEBUG)
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_format)


# input_file = "/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1_postprocessed.n5"
# output_file = (
#     "/nrs/cellmap/rhoadesj/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5"
# )
# dataset = "mito/postprocessed_mito"
# out_dataset = "mito/clean_instance_0"

input_file = "/nrs/cellmap/pattonw/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5"
output_file = (
    "/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1_postprocessed.zarr"
)
dataset = "/predictions/2023-04-10/ecs/best"
out_dataset = "ecs/postprocessed_ecs_v2"


context_padding = 10

# hole_area_threshold = 100
mask_settings = (
        "/nrs/cellmap/data/jrc_mus-liver-zon-1/staging/jrc_mus-liver-zon-1_prediction_edits.zarr",
        "s0",
        "background",
        # None,
        cube(10),
    )
    # (
    #     "/nrs/cellmap/ackermand/cellmap/jrc_mus-liver-zon-1.n5",
    #     "ld",
    #     "foreground",
    #     cube(2),
    # ),
    # (
    #     "/nrs/cellmap/rhoadesj/tmp_data/jrc_mus-liver-zon-1.n5",
    #     "volumes/predictions/cells",
    #     "background",
    #     # cube(10),
    #     cube(3),
    # ),
# ]
bad_id_ratio = 0.01

# =========
array_in = open_ds(input_file, dataset)
voxel_size = array_in.voxel_size
block_size = np.array(array_in.data.chunks) * np.array(voxel_size)
write_size = daisy.Coordinate(block_size)
context = daisy.Coordinate(np.array(voxel_size) * context_padding)
write_roi = daisy.Roi((0,) * len(write_size), write_size)
read_roi = write_roi.grow(context, context)
total_roi = array_in.roi
# string_roi = "[200000:205000,100000:110000,10000:20000]"
# parsed_start, parsed_end = zip(
#             *[
#                 tuple(int(coord) for coord in axis.split(":"))
#                 for axis in string_roi.strip("[]").split(",")
#             ]
#         )
# total_roi = daisy.Roi(
#             daisy.Coordinate(parsed_start),
#             daisy.Coordinate(parsed_end) - daisy.Coordinate(parsed_start),
#         )
# total_roi = daisy.Roi((172800, 38400, 44800), voxel_size * 1024)
# total_roi = daisy.Roi(
#     voxel_size * daisy.Coordinate(17329, 9978, 7708), voxel_size * 2048
# )

num_voxels_in_block = (read_roi / array_in.voxel_size).size
# masks = []
mask_file, mask_dataset, mask_type, structure_element = mask_settings
mito_mask = open_ds(mask_file, mask_dataset)
    # masks.append((mask, mask_type, structure_element))


def segment_function(array_in,roi):
    threshold = 0.5  # 0.5 # 0.54
# min_size = 2e7 / (8**3)
    gaussian_kernel = 2
    instance = array_in.to_ndarray(roi, fill_value=0)
    instance = gaussian(instance, sigma=gaussian_kernel)

    # # threshold
    instance = (instance > threshold).astype(np.int64)

    # # Smooth instance
    

    # make mask of unwanted object class overlaps
    # all_mask = np.zeros(instance.shape, dtype=np.uint8)
    # for mask, mask_type, structure_element in masks:
    this_read_roi = roi.snap_to_grid(mito_mask.voxel_size, mode="grow")
    this_mask = mito_mask.to_ndarray(this_read_roi, fill_value=0) 

    # RESIZE MASK TO MATCH VOXEL SIZE THEN CROP TO MATCH BLOCK
    resampled_shape = this_read_roi.shape / voxel_size
    this_mask = resize(
            this_mask,
            resampled_shape,
            order=0,
            anti_aliasing=False,
            mode="constant",
            cval=0,
            clip=True,
            preserve_range=True,
        )
        

    # if structure_element is not None:
    this_mask = erosion(this_mask, structure_element)
    crop_width = (
        (np.array(this_mask.shape) - np.array(instance.shape)) // 2
    ).astype(int)
    crop_width = [[cw, cw] for cw in crop_width]
    this_mask = skimage.util.crop(this_mask, crop_width)
    # all_mask[this_mask] = 1
    # print(f"done making mask, shape: {all_mask.shape}")

    # Find ids of objects that overlap with unwanted object classes
    # bad_ids = instance[all_mask > 0]
    # bad_ids = np.unique(bad_ids)

    # Set bad ids to 0
    # for id in bad_ids:
    #     ratio = np.sum(all_mask[instance == id]) / np.sum(instance == id)
    #     if ratio > bad_id_ratio:
    #         instance[instance == id] = 0
    #     # instance[instance == id] = 0
    # print("done bad ids, elimated %d ids" % len(bad_ids))
    # instance[instance > 0] = this_mask[instance > 0]

    instance[this_mask > 0] = 0
    print("done masking")

    # # Remove small objects
    # instance = skimage.morphology.remove_small_objects(instance, min_size=int(min_size))
    # for id in np.unique(instance):
    #     if np.sum(instance == id) < min_size:
    #         instance[instance == id] = 0
    # print("done remove small")

    # Relabel objects to smallest range of integers
    # instance = skimage.measure.label(instance)
    print("done relabel")

    # # dilate result
    # structure_element = ball(3)
    # instance = dilation(instance, structure_element)
    # print("done dilate")

    return instance
