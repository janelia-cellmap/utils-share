from scipy import ndimage
import skimage.measure
import skimage.filters
import skimage.morphology
import skimage.util
from skimage.transform import resize
import numpy as np
from skimage.segmentation import watershed
import zarr
from skimage.morphology import dilation, cube, remove_small_holes, ball
from funlib.persistence import open_ds, prepare_ds
import daisy

input_file = "/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1_postprocessed.n5"
output_file = (
    "/nrs/cellmap/rhoadesj/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5"
)
dataset = "mito/postprocessed_mito"
out_dataset = "mito/clean_instance_0"

context_padding = 50
threshold = 0.58  # 0.5 # 0.54
min_size = 2e7 / (8**3)
gaussian_kernel = 2
hole_area_threshold = 100
mask_settings = [
    (
        "/nrs/cellmap/ackermand/cellmap/jrc_mus-liver-zon-1.n5",
        "nucleus",
        "foreground",
        cube(2),
    ),
    (
        "/nrs/cellmap/ackermand/cellmap/jrc_mus-liver-zon-1.n5",
        "ld",
        "foreground",
        cube(2),
    ),
    # (
    #     "/nrs/cellmap/rhoadesj/tmp_data/jrc_mus-liver-zon-1.n5",
    #     "volumes/predictions/cells",
    #     "background",
    #     # cube(10),
    #     cube(3),
    # ),
]
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
# total_roi = daisy.Roi((172800, 38400, 44800), voxel_size * 1024)
# total_roi = daisy.Roi(
#     voxel_size * daisy.Coordinate(17329, 9978, 7708), voxel_size * 2048
# )

num_voxels_in_block = (read_roi / array_in.voxel_size).size
masks = []
for mask_file, mask_dataset, mask_type, structure_element in mask_settings:
    mask = open_ds(mask_file, mask_dataset)
    masks.append((mask, mask_type, structure_element))


def segment_function(block):
    instance = array_in.to_ndarray(block.read_roi, fill_value=0).astype(np.int64)

    # make mask of unwanted object class overlaps
    all_mask = np.zeros(instance.shape, dtype=np.uint8)
    for mask, mask_type, structure_element in masks:
        this_read_roi = block.read_roi.snap_to_grid(mask.voxel_size, mode="grow")
        if mask_type == "foreground":
            this_mask = mask.to_ndarray(this_read_roi, fill_value=0) > 0
        elif mask_type == "background":
            this_mask = mask.to_ndarray(this_read_roi, fill_value=0) == 0
        else:
            raise ValueError(f"Unknown mask type {mask_type}")
        # RESIZE MASK TO MATCH VOXEL SIZE THEN CROP TO MATCH BLOCK
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
        this_mask = dilation(this_mask, structure_element)
        crop_width = (
            (np.array(this_mask.shape) - np.array(instance.shape)) // 2
        ).astype(int)
        crop_width = [[cw, cw] for cw in crop_width]
        this_mask = skimage.util.crop(this_mask, crop_width)
        all_mask[this_mask] = 1
    print(f"done making mask, shape: {all_mask.shape}")

    # Find ids of objects that overlap with unwanted object classes
    bad_ids = instance[all_mask > 0]
    bad_ids = np.unique(bad_ids)

    # Set bad ids to 0
    for id in bad_ids:
        ratio = np.sum(all_mask[instance == id]) / np.sum(instance == id)
        if ratio > bad_id_ratio:
            instance[instance == id] = 0
        # instance[instance == id] = 0
    print("done bad ids, elimated %d ids" % len(bad_ids))

    instance[all_mask > 0] = 0
    print("done masking")

    # # Remove small objects
    # instance = skimage.morphology.remove_small_objects(instance, min_size=int(min_size))
    for id in np.unique(instance):
        if np.sum(instance == id) < min_size:
            instance[instance == id] = 0
    print("done remove small")

    # Relabel objects to smallest range of integers
    instance = skimage.measure.label(instance)
    print("done relabel")

    # # dilate result
    # structure_element = ball(3)
    # instance = dilation(instance, structure_element)
    # print("done dilate")

    return instance.astype(np.uint64)
