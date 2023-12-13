import json
import multiprocessing
# workaround for MacOS:
# this needs to be set before importing any library that uses multiprocessing
# multiprocessing.set_start_method('fork')
import logging
import os
import daisy
import subprocess
from funlib.persistence import open_ds, prepare_ds
import numpy as np


logging.basicConfig(level=logging.INFO)


def run_scheduler(
    total_roi,
    read_roi,
    write_roi,
    num_workers
):

    task = daisy.Task(
        "mito_clean",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=start_worker,
        num_workers=num_workers,
        read_write_conflict=False,
        fit="shrink",
    )

    daisy.run_blockwise([task])


def start_worker():

    worker_id = daisy.Context.from_env()["worker_id"]
    task_id = daisy.Context.from_env()["task_id"]

    print(f"worker {worker_id} started for task {task_id}...")

    subprocess.run(["python", "./worker.py"])

    subprocess.run([ "bsub",
                    "-I",
                    "-P",
                    "cellmap",
                    "-J",
                    "mito_pred",
                    "-n",
                    "5",
                    "python", "./worker.py"])

    # do the same on a cluster node:
    # num_cpus_per_worker = 4
    # subprocess.run(["bsub", "-I", f"-n {num_cpus_per_worker}", "python", "./worker.py"])


if __name__ == "__main__":
    # this is so insecure you might as well call it Donald
    from config import *

    array_out = prepare_ds(output_file,
								out_dataset,
								array_in.roi,
								voxel_size = voxel_size,
								write_size= block_size_nm,
								dtype = np.uint64)
    run_scheduler(
        array_in.roi,
        read_roi=block_size_nm.grow(context_nm, context_nm),
        write_roi=block_size_nm,
        num_workers=100
    )