import logging
import daisy
import subprocess
import yaml
from funlib.persistence import open_ds, prepare_ds
import numpy as np
logging.basicConfig(level=logging.INFO)

def start_worker():
    num_cpus_per_worker = 10
    subprocess.run(["bsub","-P","cellmap","-J","worker","-n",str(num_cpus_per_worker), "python", "worker.py"])

def run_scheduler(
    total_roi,
    read_roi,
    write_roi,
    num_workers
):
    dummy_task = daisy.Task(
        "dummy_task",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=start_worker,
        num_workers=num_workers,
        check_function=None,
        read_write_conflict=True,
        fit="overhang",
        max_retries=2,
        timeout=None,
    )
    daisy.run_blockwise([dummy_task])

if __name__ == "__main__":
    config_path = "config.yaml"
    configs = yaml.load(open(config_path), Loader=yaml.FullLoader)
    input_file = configs["input_file"]
    dataset = configs["dataset"]
    output_file = configs["output_file"]
    out_dataset = configs["out_dataset"]
    
    
    array_in = open_ds(input_file, dataset)

    chunks = array_in.data.chunks
    voxel_size = array_in.voxel_size
    block_size_nm = [chunks[0]*voxel_size[0],  chunks[1]*voxel_size[1], chunks[2]*voxel_size[2]]
    context_nm = (50*voxel_size[0],50*voxel_size[1],50*voxel_size[2]) #50 pixel overlap

    print("block_size_nm", block_size_nm)
    print("context_nm", context_nm)
    print("voxel_size", voxel_size)
    print("chunks", chunks)
    
    array_out = prepare_ds(output_file,
								out_dataset,
								array_in.roi,
								voxel_size = array_in.voxel_size,
								write_size= block_size_nm,
								dtype = np.uint64)

    
    write_size = daisy.Coordinate(block_size_nm)
    write_roi = daisy.Roi((0,) * len(write_size), write_size)
    print("write_roi" + str (write_roi))
    read_roi = write_roi.grow(context_nm, context_nm)
    total_roi = array_in.roi.grow(context_nm, context_nm)
    run_scheduler(
        total_roi,
        read_roi,
        write_roi,
        num_workers=100,
    )