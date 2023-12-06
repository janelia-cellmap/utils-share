# %%
from glob import glob
import os
import daisy
from funlib.persistence import open_ds, Array, prepare_ds
import torch
import GPUtil
import numpy as np
import funlib.segment as seg

# import multiprocessing

# multiprocessing.set_start_method("spawn", force=True)

# ================================================================================================
# SETUP
num_workers = 12
dataset = "jrc_mus-liver-zon-1"
in_path_base = "/nrs/cellmap/data/{dataset}".format(dataset=dataset)
in_path = "{in_path_base}/empanada/predictions.zarr".format(in_path_base=in_path_base)
# create output dataset
out_path = "{in_path_base}/empanada/predictions_full.zarr".format(
    in_path_base=in_path_base
)
# get original dataset info
original_name = "recon-1/em/fibsem-uint8/s1"
original_ds = open_ds(
    "/nrs/cellmap/data/{dataset}/{dataset}.zarr".format(dataset=dataset),
    original_name,
    mode="r",
)

name = "mito_chunk_{id}"
out_name = "mito_pred"
overlap_threshold = 0.75
# chunk_shape = (8192, 8192, 8192)
chunk_shape = (4096, 4096, 4096)

# context = daisy.Coordinate((512, 512, 512)) * voxel_size

# ================================================================================================
# for testing
original_ds = open_ds(
    "/nrs/cellmap/rhoadesj/tmp_data/jrc_mus-liver-zon-2.zarr",
    "/volumes/crop356/raw/",
    mode="r",
)
in_path = "/nrs/cellmap/rhoadesj/tmp_data/empanada/predictions.zarr"
out_path = "/nrs/cellmap/rhoadesj/tmp_data/empanada/test_full.zarr"
chunk_shape = (800, 800, 600)
# ================================================================================================

total_roi = original_ds.roi
voxel_size = original_ds.voxel_size
chunk_size = daisy.Coordinate(chunk_shape) * voxel_size

# get chunk ids (assumes chunk ids are integers at end of name)
chunk_ids = [
    int(f.replace(in_path + os.sep, "").split("_")[-1])
    for f in glob(os.path.join(in_path, name.format(id="*")))
]

# open datasets
srcs = []
for id in chunk_ids:
    in_name = name.format(id=id)
    srcs.append(open_ds(in_path, in_name, mode="r"))

# try:
#     out = open_ds(out_path, out_name, mode="r+")
# except:
out = prepare_ds(
    out_path,
    out_name,
    total_roi=total_roi,
    voxel_size=voxel_size,
    dtype=np.uint64,
    delete=True,
)

# device = get_best_gpu()


def get_best_gpu():
    return torch.device("cpu")
    if torch.cuda.is_available():
        # Get the GPU details
        GPUs = GPUtil.getGPUs()
        # Find out the GPU with maximum available memory
        GPUs = sorted(GPUs, key=lambda x: x.memoryFree)
        best_GPU = GPUs[-1]
        print(f"Number of GPUs available: {len(GPUs)}")
        print(f"GPU with most free memory: {best_GPU.id}")
        return torch.device(f"cuda:{best_GPU.id}")
    else:
        print("No GPU available.")
        return torch.device("cpu")


def remap_ids(data, id_start=1, device=get_best_gpu()):
    with torch.no_grad():
        # 1) find each src's unique IDs
        src_ids = torch.unique(data).to(device)

        # 2) remap each src's IDs to a unique range for each src, except for id==0
        new_ids = torch.zeros(src_ids.shape, dtype=torch.int64).to(device)
        for i, id in enumerate(src_ids):
            if id == 0:
                continue
            data[data == id] = id_start + i
            new_ids[i] = id_start + i
        id_start += len(src_ids)

    return data, new_ids, id_start


def unite_overlap(
    src1, src2, overlap_threshold=overlap_threshold, device=get_best_gpu()
):
    with torch.no_grad():
        # 1) get unique remapped IDs for each src
        src1, src1_new_ids, id_start = remap_ids(src1)
        src2, src2_new_ids, id_start = remap_ids(src2, id_start)

        # 2) combine srcs into one array
        data = torch.stack([src1, src2]).to(device)

        # 3) find IDs that overlap between srcs, and get the number of pixels they overlap
        # new_ids = torch.unique(data.flatten()).to(device)
        overlaps = torch.zeros((len(src1_new_ids), len(src2_new_ids))).to(device)
        # overlaps = torch.zeros((len(new_ids), len(new_ids))).to(device)
        min_sizes = torch.zeros_like(overlaps).to(device)
        sizes = torch.bincount(data.flatten()).to(device)
        for i, src1_id in enumerate(src1_new_ids):
            if src1_id == 0:
                continue
            # get overlapping IDs
            src2_ids = torch.unique(src2[src1 == src1_id]).to(device)
            # get overlap counts
            for j, src2_id in enumerate(src2_ids):
                if src2_id == 0:
                    continue
                overlap = torch.sum((src1 == src1_id) & (src2 == src2_id)).to(device)
                overlaps[i, j] += overlap
                min_sizes[i, j] += torch.min(sizes[src1_id], sizes[src2_id]).to(device)

        # 4) find IDs that overlap greater than overlap_threshold
        overlaps = (overlaps / min_sizes) > overlap_threshold

        # 5) remap overlapping IDs with the sufficient overlap to the same ID in the smallest possible range
        result = torch.zeros_like(src1).to(device)
        final_id = 1
        for x, y in torch.nonzero(overlaps):
            i = src1_new_ids[x]
            j = src2_new_ids[y]
            mask = (src1 == i) | (src2 == j)
            result[mask] = final_id
            final_id += 1
    return result


def unite_and_write(
    src1, src2, out, roi, overlap_threshold=overlap_threshold, device=get_best_gpu()
):
    src1 = torch.as_tensor(src1.to_ndarray(roi).astype(np.int32), device=device)
    src2 = torch.as_tensor(src2.to_ndarray(roi).astype(np.int32), device=device)
    new_src = unite_overlap(
        src1, src2, overlap_threshold=overlap_threshold, device=device
    )
    final = unite_overlap(
        new_src,
        torch.as_tensor(out.to_ndarray(roi).astype(np.int32), device=device),
        overlap_threshold=overlap_threshold,
        device=device,
    )
    out[roi] = final.cpu().numpy().astype(np.uint64)


# make worker function
def process_func(block: daisy.Block):
    with torch.no_grad():
        block_id = block.block_id
        print(f"Processing block {block_id}")
        read_roi = block.read_roi

        # collect relevant sources and write to output
        these_srcs = []
        for src in srcs:
            if src.roi.intersects(read_roi):
                these_srcs.append(src)
        if len(these_srcs) == 0:
            print(f"No source data found for block {block_id}")
            return
        else:
            print(f"Found {len(these_srcs)} sources for block {block_id}")

        # iterate over overlapping chunks
        for i in range(len(these_srcs)):
            for j in range(i + 1, len(these_srcs)):
                src1 = these_srcs[i]
                src2 = these_srcs[j]
                roi = read_roi.intersect(src1.roi)
                if not roi.intersects(src2.roi):
                    continue
                roi = roi.intersect(src2.roi)
                unite_and_write(
                    src1,
                    src2,
                    out,
                    roi,
                    overlap_threshold=overlap_threshold,
                    # device=device,
                )


# %%

# setup daisy task
block_read_roi = block_write_roi = daisy.Roi((0, 0, 0), chunk_size)
# block_read_roi = block_read_roi.grow(context, context)
task = daisy.Task(
    task_id="unite",
    # total_roi=total_roi.grow(context, context),
    total_roi=total_roi,
    read_roi=block_write_roi,
    write_roi=block_read_roi,
    process_function=process_func,
    num_workers=num_workers,
    read_write_conflict=True,
    fit="overhang",
)

# run daisy task
daisy.run_blockwise([task])

# %%
