import subprocess

import daisy
import glob
import logging
import numpy as np
import os
import tempfile
from config import *

logger = logging.getLogger(__name__)


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

    # define in config.py
    # write_size = daisy.Coordinate(block_size)
    # write_roi = daisy.Roi((0,) * len(write_size), write_size)
    # read_roi = write_roi.grow(context, context)
    # total_roi = array_in.roi.grow(context, context)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"total_roi: {total_roi}:")
        print(f"read_roi: {read_roi}:")
        print(f"write_roi: {write_roi}:")

        def start_worker():
            worker_id = daisy.Context.from_env()["worker_id"]
            task_id = daisy.Context.from_env()["task_id"]

            logger.info(f"worker {worker_id} started for task {task_id}...")

            # subprocess.run(["python", "./segment_worker.py"])

            # do the same on a cluster node:
            # num_cpus_per_worker = 4
            subprocess.run(
                [
                    "bsub",
                    "-I",
                    "-P",
                    "cellmap",
                    "-J",
                    "segment_worker",
                    "-n",
                    "5",
                    "python",
                    tmpdir,
                    "./segment_worker.py",
                ]
            )

        task = daisy.Task(
            "segment_blockwise",
            total_roi,
            read_roi,
            write_roi,
            process_function=start_worker,
            num_workers=num_workers,
            fit="shrink",
            # read_write_conflict=True,
        )
        daisy.run_blockwise([task])

        def start_worker():
            worker_id = daisy.Context.from_env()["worker_id"]
            task_id = daisy.Context.from_env()["task_id"]

            logger.info(f"worker {worker_id} started for task {task_id}...")

            # subprocess.run(["python", "./segment_worker.py"])

            # do the same on a cluster node:
            # num_cpus_per_worker = 4
            subprocess.run(
                [
                    "bsub",
                    "-I",
                    "-P",
                    "cellmap",
                    "-J",
                    "relabel_worker",
                    "-n",
                    "5",
                    "python",
                    tmpdir,
                    "./relabel_worker.py",
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
        )

        daisy.run_blockwise([task])


if __name__ == "__main__":
    segment_blockwise()
