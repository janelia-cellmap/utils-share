from glob import glob
import os
import daisy
from funlib.persistence import open_ds, Array
import torch
import numpy as np

# ================================================================================================
# SETUP
num_workers = 128
dataset = "jrc_mus-liver-zon-1"
in_path_base = "/nrs/cellmap/data/{dataset}".format(dataset=dataset)
in_path = "{in_path_base}/empanada/predictions_chunk_{id}.zarr".format(
    in_path_base=in_path_base, id="{id}"
)
# create output dataset
out_path = "{in_path_base}/empanada/predictions_full.zarr".format(
    in_path_base=in_path_base
)
# get original dataset info
original_name = "recon-1/em/fibsem-uint8/s1"
original_ds = open_ds(
    "/nrs/cellmap/data/{dataset}/{dataset}.zarr".format(dataset=dataset), original_name
)

name = "mito_pred"
overlap_threshold = 0.75
# chunk_shape = (8192, 8192, 8192)
chunk_shape = (4096, 4096, 4096)
chunk_size = daisy.Coordinate(chunk_shape) * voxel_size

context = daisy.Coordinate((420, 420, 420)) * voxel_size

# ================================================================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# get chunk ids
chunk_ids = [int(f.split(".")[0].split("_")[-1]) for f in glob(in_path.format(id="*"))]

# open datasets
srcs = [open_ds(in_path.format(id=i), name) for i in chunk_ids]

out_name = name
out = open_ds(out_path, out_name, mode="w", dtype=np.uint64, shape=srcs[0].shape)

srcs.append(out)

# get original dataset info
total_roi = original_ds.roi
voxel_size = original_ds.voxel_size


# make worker function
def process_func(block: daisy.Block):
    block_id = block.block_id
    print(f"Processing block {block_id}")
    block_roi = block.read_roi.grow(context, context)

    # collect relevant sources
    these_srcs = []
    for src in srcs:
        if src.roi.intersects(block_roi):
            these_srcs.append(src)
    if len(these_srcs) == 0:
        print(f"No source data found for block {block_id}")
        return
    else:
        print(f"Found {len(these_srcs)} sources for block {block_id}")

    # get intersecting roi
    read_roi = block_roi
    for src in these_srcs:
        read_roi = read_roi.intersect(src.roi)
    write_roi = read_roi.intersect(block.write_roi)

    # read data
    data = torch.stack([src[read_roi] for src in these_srcs]).requires_grad_(False)
    data = data.to(device)

    # process data
    # 1) find each src's unique IDs
    srcs_ids = []
    for i in range(data.shape[0]):
        srcs_ids.append(torch.unique(data[i]).requires_grad_(False)).to(device)

    # 2) remap each src's IDs to a unique range for each src, except for id==0
    id_start = 1
    new_ids = []
    for i, src_ids in enumerate(srcs_ids):
        for j, id in enumerate(src_ids):
            if id == 0:
                continue
            data[i][data[i] == id] = id_start + j
        new_ids.append(
            torch.arange(id_start, id_start + j + 1).requires_grad_(False).to(device)
        )
        id_start += j + 1
    all_new_ids = torch.arange(id_start).requires_grad_(False).to(device)

    # 3) find IDs that overlap between srcs, and get the number of pixels they overlap
    overlaps = (
        torch.zeros((id_start, id_start), dtype=torch.uint64)
        .requires_grad_(False)
        .to(device)
    )
    min_sizes = (
        torch.zeros((id_start, id_start), dtype=torch.uint64)
        .requires_grad_(False)
        .to(device)
    )
    sizes = torch.bincount(data.flatten()).requires_grad_(False).to(device)
    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            for i_id in new_ids[i]:
                if i_id == 0:
                    continue
                # get overlapping IDs
                j_ids = torch.unique(data[j][data[i] == i_id])
                # get overlap counts
                for j_id in j_ids:
                    if j_id == 0:
                        continue
                    overlap = torch.sum((data[i] == i_id) & (data[j] == j_id))
                    overlaps[i, j] += overlap
                    min_sizes[i, j] += torch.min(sizes[i_id], sizes[j_id])
                    # overlaps[j, i] += overlap

    # 4) find IDs that overlap greater than overlap_threshold
    # overlaps[overlaps > 0] /= min_sizes[overlaps > 0]
    # overlaps[overlaps < overlap_threshold] = 0
    overlaps = (overlaps / min_sizes) > overlap_threshold
    overlap_inds = torch.nonzero(overlaps)

    # 5) remap overlapping IDs with the sufficient overlap to the same ID in the smallest possible range
    result = (
        torch.zeros(data.shape[1:], dtype=torch.uint64).requires_grad_(False).to(device)
    )
    final_id = 1
    for x, y in overlap_inds:
        mask = (
            torch.max((data == x) | (data == y), dim=0).requires_grad_(False).to(device)
        )
        result[mask] = final_id
        final_id += 1

    # 6) adjust IDs to be unique across blocks
    result[result > 0] += (block_id * torch.prod(write_roi.shape)).astype(torch.uint64)

    # 7) write data
    result = Array(result.cpu().numpy(), read_roi)
    out[write_roi] = result[write_roi]


# setup daisy task
block_read_roi = block_write_roi = daisy.Roi((0, 0, 0), chunk_size)
task = daisy.Task(
    task_id="unite",
    total_roi=total_roi,
    read_roi=block_read_roi,
    write_roi=block_write_roi,
    process_function=process_func,
    num_workers=1,
    read_write_conflict=True,
    fit="overhang",
)

# run daisy task
daisy.run_blockwise([task])
