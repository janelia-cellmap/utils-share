import skimage.measure
import skimage.filters
import skimage.morphology
from funlib.segment.arrays import segment_blockwise
from funlib.geometry import Coordinate
from funlib.persistence import open_ds, prepare_ds
import numpy as np


def get_clean_instance_peroxisomes(
    peroxi_dist,
    LD,
    mito,
    ECS,
    epithelial,
    hepatocyte,
    threshold=0.5,
    min_size=5e6,
    gaussian_kernel=5,
):
    # gaussian smooth distance predictions
    peroxi_dist = skimage.filters.gaussian(peroxi_dist, gaussian_kernel)
    # threshold precictions
    binary_peroxi = peroxi_dist > threshold
    # get instance labels
    instance_peroxi = skimage.measure.label(binary_peroxi)[0].astype(np.uint64)
    # relabel background to 0
    instance_peroxi[binary_peroxi == 0] = 0
    # make mask of unwanted object class overlaps
    all_mask = LD | mito | ECS | epithelial | hepatocyte == 0
    # find ids of peroxisomes that overlap with unwanted object classes
    bad_ids = np.unique(instance_peroxi[all_mask])
    # set bad ids to 0
    for id in bad_ids:
        instance_peroxi[instance_peroxi == id] = 0
    # remove small objects
    instance_peroxi = skimage.morphology.remove_small_objects(
        instance_peroxi, min_size=min_size
    )
    # relabel objects to smallest range of integers
    instance_peroxi = skimage.measure.label(instance_peroxi)
    return instance_peroxi.astype(np.uint64)


def get_seg_func(mask_out_arrays, **kwargs):
    return lambda array_in, roi: get_clean_instance_peroxisomes(
        array_in.to_ndarray(roi),
        *[mask_out_array.to_ndarray(roi) for mask_out_array in mask_out_arrays],
        **kwargs
    )


# setup options here
in_path = ...
in_name = ...
mask_paths = [...]
mask_names = [...]
out_path = ...
out_name = ...
threshold = 0.5
min_size = 5e6
gaussian_kernel = ...
daisy_num_workers = 8
block_size = Coordinate(block_size)
context = Coordinate(context)


# ===============================================================================
array_in = open_ds(in_path, in_name, mode="r")
mask_out_arrays = [
    open_ds(mask_path, mask_name, mode="r")
    for mask_path, mask_name in zip(mask_paths, mask_names)
]
array_out = prepare_ds(
    out_path,
    out_name,
    total_roi=array_in.roi,
    voxel_size=array_in.voxel_size,
    dtype=np.uint64,
    delete=True,
)

segment_blockwise(
    array_in,
    array_out,
    block_size * array_in.voxel_size,
    context * array_in.voxel_size,
    daisy_num_workers,
    get_seg_func(
        mask_out_arrays,
        threshold=threshold,
        min_size=min_size,
        gaussian_kernel=gaussian_kernel,
    ),
)
