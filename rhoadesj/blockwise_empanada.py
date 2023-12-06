# %%
import numpy as np
from funlib.segment.arrays import segment_blockwise
from funlib.persistence import open_ds, prepare_ds
from block_inference3d import get_blockwise_empanada
from funlib.geometry import Coordinate, Roi


# ================================================================================================
# SETUP
dataset = "jrc_mus-liver-zon-1"
in_path_base = "/nrs/cellmap/data/{dataset}".format(dataset=dataset)
in_path = "{in_path_base}/{dataset}.zarr".format(
    in_path_base=in_path_base, dataset=dataset
)
in_name = "recon-1/em/fibsem-uint8/s1"
out_path = "{in_path_base}/empanada/predictions_full.zarr".format(
    in_path_base=in_path_base
)
out_name = "mito_pred"

block_size = (4096, 4096, 4096)
context = (512, 512, 512)

daisy_num_workers = 1

blockwise_kwargs = {
    "config": "/nrs/cellmap/rhoadesj/empanada/projects/liver-zonation/liver.yaml",
    "num_workers": 128,
    "fine_boundaries": True,
    "qlen": 11,
    "nms_kernel": 21,
    "seg_thr": 0.9,
    "nms_thr": 0.25,
    "pixel_vote_thr": 1,
    "min_size": 10000,
    "min_span": 50,
    # "nmax": 2000000,
}


# ================================================================================================
# # TESTING SETUP
# dataset = "jrc_mus-liver-zon-2"
# in_path_base = "/nrs/cellmap/rhoadesj/tmp_data/"
# in_path = "{in_path_base}/{dataset}.zarr".format(
#     in_path_base=in_path_base, dataset=dataset
# )
# in_name = "/volumes/crop356/raw/"
# out_path = "{in_path_base}/empanada/predictions_full.zarr".format(
#     in_path_base=in_path_base
# )
# out_name = "mito_pred"

# block_size = (800, 800, 600)
# context = (0, 0, 200)

# blockwise_kwargs = {
#     "config": "/nrs/cellmap/rhoadesj/empanada/projects/liver-zonation/liver.yaml",
#     "num_workers": 128,
#     "fine_boundaries": True,
#     "qlen": 11,
#     "nms_kernel": 21,
#     "seg_thr": 0.9,
#     "nms_thr": 0.25,
#     "pixel_vote_thr": 1,
#     "min_size": 10000,
#     "min_span": 50,
#     "downsample_f": 2,
#     # "nmax": 2000000,
# }
# # ================================================================================================
block_size = Coordinate(block_size)
context = Coordinate(context)
array_in = open_ds(in_path, in_name, mode="r")
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
    get_blockwise_empanada(**blockwise_kwargs),
)

# %%
