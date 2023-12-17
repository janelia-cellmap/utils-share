import skimage.measure
import skimage.filters
import skimage.morphology
import numpy as np
from skimage.segmentation import watershed
import zarr
from scipy.ndimage import binary_fill_holes
from skimage.morphology import dilation, cube
from funlib.persistence import open_ds, prepare_ds
import daisy


input_file = (
    "/nrs/cellmap/rhoadesj/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5"
)
output_file = (
    "/nrs/cellmap/rhoadesj/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5"
)
dataset = "peroxisome/instance"
out_dataset = "peroxisome/clean_instance_0"

context_padding = 10
threshold = 0.5
min_size = 7e6 / (8**3)
gaussian_kernel = 2

mask_settings = [
    ("/nrs/ackermand/cellmap/jrc_mus-liver-zon-1.n5", "nucleus", "foreground", cube(2)),
    ("/nrs/ackermand/cellmap/jrc_mus-liver-zon-1.n5", "ld", "foreground", cube(1)),
    ("/nrs/cellmap/data/jrc_mus-liver-zon-1.n5", "cell", "background", cube(3)),
    ("/nrs/cellmap/data/jrc_mus-liver-zon-1.n5", "mito", "foreground", cube(2)),
]

# =========
array_in = open_ds(input_file, dataset)
masks = []
for mask_file, mask_dataset, mask_type in mask_settings:
    mask = open_ds(mask_file, mask_dataset)
    masks.append((mask, mask_type))
voxel_size = array_in.voxel_size
block_size = np.array(array_in.data.chunks) * np.array(voxel_size)
write_size = daisy.Coordinate(block_size)
try:
    array_out = open_ds(output_file, out_dataset, mode="a")
except KeyError:
    array_out = prepare_ds(
        output_file,
        out_dataset,
        array_in.roi,
        voxel_size=voxel_size,
        write_size=write_size,
        dtype=np.uint64,
    )
array_out = open_ds(output_file, out_dataset, mode="a")
context = daisy.Coordinate(np.array(voxel_size) * context_padding)

write_roi = daisy.Roi((0,) * len(write_size), write_size)
read_roi = write_roi.grow(context, context)
total_roi = array_in.roi.grow(context, context)

num_voxels_in_block = (read_roi / array_in.voxel_size).size


def segment_function(block):
    instance = array_in.to_ndarray(block.read_roi, fill_value=0)

    # make mask of unwanted object class overlaps
    all_mask = np.zeros_like(instance, dtype=np.uint8)
    for mask, mask_type, structure_element in masks:
        if mask_type == "foreground":
            this_mask = mask.to_ndarray(block.read_roi, fill_value=0) > 0
        elif mask_type == "background":
            this_mask = mask.to_ndarray(block.read_roi, fill_value=0) == 0
        else:
            raise ValueError(f"Unknown mask type {mask_type}")
        this_mask = dilation(this_mask, structure_element)
        all_mask[this_mask] = 1

    print("done mask")
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
    # remove small objects
    instance = skimage.morphology.remove_small_objects(
        instance, min_size=int(np.ceil(min_size))
    )
    print("done remove small")
    # relabel objects to smallest range of integers
    instance = skimage.measure.label(instance)
    print("done relabel")
    # print("done relabel", instance.shape)

    return instance.astype(np.uint64)
