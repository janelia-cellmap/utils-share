import os
import subprocess
import time

import daisy
import numpy as np
import tempfile
from funlib.persistence import prepare_ds
from config import *


def segment_blockwise():
    """Segment an array in parallel.

    Args:

        array_in (``Array``):

            The input data needed by `segment_function` to produce a
            segmentation.

        array_out (``Array``):

            The array to write to. Should initially be empty (i.e., all zeros).

        block_size (``daisy.Coordinate``):

            The size of the blocks to segment (without context), in world
            units.

        context (``daisy.Coordinate``):

            The amount of padding to add (in the negative and postive
            direction) for context, in world units.

        num_workers (``int``):

            The number of workers to use.

        segment_function (function):

            A function taking arguments ``array_in`` and ``roi`` to produce a
            segmentation for ``roi`` only. Expected to return an ndarray of the
            shape of ``roi`` (using the voxel size of ``array_in``) with
            datatype ``np.uint64``. Zero in the segmentation are considered
            background and will stay zero.
    """
    global total_roi, read_roi, write_roi, tmp_prefix, num_workers, num_cpus, log_file_path, context
    if os.path.exists(os.path.join(output_file, out_dataset)):
        print("Output dataset already exists.")
        for i in range(10):
            print(f"Deleting in {10-i} seconds...")
            time.sleep(1)
        os.system(f"rm -rf {output_file}/{out_dataset}")
    array_out = prepare_ds(
        output_file,
        out_dataset,
        total_roi,
        voxel_size=voxel_size,
        write_size=write_size,
        dtype=np.uint64,
    )

    print("Starting segmentation...")
    os.system(f"ulimit -n {10 * num_workers}")
    os.makedirs(tmp_prefix, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdir:
        print(f"total_roi: {total_roi.grow(context, context)}:")
        print(f"read_roi: {read_roi}:")
        print(f"write_roi: {write_roi}:")

        def start_worker():
            worker_id = daisy.Context.from_env()["worker_id"]
            task_id = daisy.Context.from_env()["task_id"]

            print(f"worker {worker_id} started for task {task_id}...")
            log_file_path = f"./daisy_logs/{task_id}/worker_{worker_id}"
            # subprocess.run(["python", "./segment_worker.py"])

            # do the same on a cluster node:
            # num_cpus_per_worker = 4
            subprocess.run(
                [
                    "bsub",
                    "-K",
                    "-P",
                    "cellmap",
                    "-J",
                    "segment_worker",
                    "-n",
                    str(num_cpus),
                    "-e",
                    f"{log_file_path}.err",
                    "-o",
                    f"{log_file_path}.out",
                    "python",
                    "./segment_worker.py",
                    tmpdir,
                ]
            )

        task = daisy.Task(
            "segment_blockwise",
            total_roi.grow(context, context),
            read_roi,
            write_roi,
            process_function=start_worker,
            num_workers=num_workers,
            fit="shrink",
            read_write_conflict=True,
            timeout=10,
        )
        daisy.run_blockwise([task])

        print("Finished segmentation. Relabeling...")
        # give a second for the fist task to finish
        time.sleep(1)
        read_roi = write_roi

        def start_worker():
            worker_id = daisy.Context.from_env()["worker_id"]
            task_id = daisy.Context.from_env()["task_id"]

            print(f"worker {worker_id} started for task {task_id}...")
            log_file_path = f"./daisy_logs/{task_id}/worker_{worker_id}"
            # subprocess.run(["python", "./segment_worker.py"])

            # do the same on a cluster node:
            # num_cpus_per_worker = 4
            subprocess.run(
                [
                    "bsub",
                    "-K",
                    "-P",
                    "cellmap",
                    "-J",
                    f"relabel_worker_{worker_id}",
                    "-n",
                    str(num_cpus),
                    "-e",
                    f"{log_file_path}.err",
                    "-o",
                    f"{log_file_path}.out",
                    "python",
                    "./relabel_worker.py",
                    tmpdir,
                ]
            )

        task = daisy.Task(
            "relabel_blockwise",
            total_roi,
            read_roi,
            write_roi,
            process_function=start_worker,
            num_workers=num_workers,
            fit="shrink",
            timeout=10,
        )

        daisy.run_blockwise([task])

    print("All done. Enjoy your segmentation.")


if __name__ == "__main__":
    os.system(f"ulimit -n {10 * num_workers}")
    segment_blockwise()
