import os
import daisy
from funlib.persistence import open_ds
from funlib.geometry import Coordinate, Roi

# ================================================================================================
# SETUP
dataset = "jrc_mus-liver-zon-1"
in_path_base = "/nrs/cellmap/data/{dataset}".format(dataset=dataset)
in_path = "{in_path_base}/{dataset}.zarr".format(
    in_path_base=in_path_base, dataset=dataset
)
in_name = "recon-1/em/fibsem-uint8/s1"

launch_str = 'bsub -n 48 -q gpu_t4 -gpu "num=1" -P cellmap -o /nrs/cellmap/rhoadesj/empanada/projects/liver-zonation/daisy/{dataset}_{id}.out -e /nrs/cellmap/rhoadesj/empanada/projects/liver-zonation/daisy/{dataset}_{id}.err python /nrs/cellmap/rhoadesj/empanada/scripts/pdl_inference3d.py /nrs/cellmap/rhoadesj/empanada/projects/liver-zonation/liver.yaml {in_path} {in_path_base}/empanada/{dataset}_predictions.zarr -data-key {in_name} -out-key mito_chunk_{id} -num_workers 96 --fine-boundaries -qlen 11 -nms-kernel 21 -seg-thr 0.8 -nms-thr 0.25 -pixel-vote-thr 1 -min-size 10000 -min-span 50 -nmax 2000000 -roi {roi}'

# chunk_shape = (8192, 8192, 8192)
# chunk_shape = (4096, 4096, 4096)
chunk_shape = (2048, 2048, 2048)
context_shape = (512, 512, 512)
# ================================================================================================
# ================================================================================================
# for testing
dataset = "jrc_mus-liver-zon-2"
in_path_base = "/nrs/cellmap/rhoadesj/tmp_data/"
in_path = "{in_path_base}/{dataset}.zarr".format(
    in_path_base=in_path_base, dataset=dataset
)
in_name = "/volumes/crop356/raw/"

launch_str = 'bsub -n 12 -q gpu_rtx -gpu "num=1" -P cellmap -o /nrs/cellmap/rhoadesj/empanada/projects/liver-zonation/daisy/test_{id}.out -e /nrs/cellmap/rhoadesj/empanada/projects/liver-zonation/daisy/test_{id}.err python /nrs/cellmap/rhoadesj/empanada/scripts/pdl_inference3d.py /nrs/cellmap/rhoadesj/empanada/projects/liver-zonation/liver.yaml {in_path} {in_path_base}/empanada/predictions.zarr -data-key {in_name} -out-key mito_chunk_{id} -num_workers 24 --fine-boundaries -qlen 11 -nms-kernel 21 -seg-thr 0.9 -nms-thr 0.25 -pixel-vote-thr 1 -min-size 10000 -min-span 50 -downsample-f 2 -nmax 2000000 -roi {roi}'

chunk_shape = (800, 800, 600)
context_shape = (0, 0, 200)


# ================================================================================================

src = open_ds(in_path, in_name, mode="r")
roi = src.roi
voxel_size = src.voxel_size

context = Coordinate(context_shape) * voxel_size
chunk_size = Coordinate(chunk_shape) * voxel_size

print("Chunk shape: ", chunk_shape)
print("Chunk size: ", chunk_size)
print("Context: ", context)

print("ROI: ", roi)


# define launch function
def launch_func(block: daisy.Block):
    # Note: currently write ROI is not used
    block_roi: Roi = block.read_roi
    block_roi = block_roi.intersect(roi)
    roi_str = ",".join([f"{s},{e}" for s, e in zip(block_roi.begin, block_roi.shape)])
    print(f"Launching block with {block_roi}")
    # launch_str_block = launch_str.format(
    #     dataset=dataset,
    #     id=block.block_id[1],
    #     in_path=in_path,
    #     in_path_base=in_path_base,
    #     in_name=in_name,
    #     roi=roi_str,
    # )
    launch_str_block = launch_str.format(
        id=block.block_id[1],
        in_path=in_path,
        in_path_base=in_path_base,
        in_name=in_name,
        roi=roi_str,
    )
    print(launch_str_block + "\n")
    os.system(launch_str_block)


# define daisy task
block_read_roi = Roi((0, 0, 0), chunk_size)
block_write_roi = block_read_roi.grow(-context, -context)

task = daisy.Task(
    task_id="launch",
    total_roi=roi,
    read_roi=block_read_roi,
    write_roi=block_write_roi,
    process_function=launch_func,
    num_workers=1,
    # read_write_conflict=True,
    read_write_conflict=False,
    fit="shrink",
)
daisy.run_blockwise([task])
