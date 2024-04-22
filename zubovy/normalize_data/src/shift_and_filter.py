import os
from typing import Tuple
from dask.distributed import wait
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster


import zarr
import time
from numcodecs.abc import Codec
import numpy as np
from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import Client
from numcodecs import Zstd
from toolz import partition_all

import click


def separate_store_path(store: str,
                        path: str):
    new_store, path_prefix = os.path.split(store)
    if ".zarr" in path_prefix or ".n5" in path_prefix:
        return store, path
    return separate_store_path(new_store, os.path.join(path_prefix, path))

def renormalize_and_save_slice(
        source: zarr.Array, 
        dest: zarr.Array, 
        out_slices: Tuple[slice, ...],
        global_min: int,
        global_max: int,
        out_data_type: str
        ):
    
    in_slices = tuple(out_slice for out_slice in out_slices)
    source_data = np.array(source[in_slices])
    #filtered_data = np.where((source_data >= global_min) & (source_data <= global_max), source_data-global_min, 0)
    filtered_data = np.where((source_data >= global_min) & (source_data <= global_max),
                             np.round((source_data-global_min)/(global_max-global_min)*255).astype(out_data_type),
                             0)
    dest[out_slices] = filtered_data
    return 1

def copy_arrays(z_src: zarr.Group | zarr.core.Array,
                dest_root: zarr.Group,
                global_min: int,
                global_max: int,
                client: Client,
                num_workers: int,
                out_data_type: str,
                comp: Codec):
    
    
    # store original array in a new .zarr file as an arr_name
    client.cluster.scale(num_workers)
    if isinstance(z_src, zarr.core.Array):
        z_arrays = [z_src]
    else:
        z_arrays = [key_val_arr[1] for key_val_arr in (z_src.arrays())]
    for src_arr in z_arrays:
        

        start_time = time.time()
        if out_data_type == '':
            out_data_type = src_arr.dtype
        dest_arr = dest_root.require_dataset(
            src_arr.name, 
            shape=src_arr.shape, 
            chunks=src_arr.chunks, 
            dtype=out_data_type, 
            compressor=src_arr.compressor, 
            dimension_separator='/')#,
            # fill_value=0,
            # exact=True)

        out_slices = slices_from_chunks(normalize_chunks(dest_arr.chunks, shape=dest_arr.shape))
        # break the slices up into batches, to make things easier for the dask scheduler
        out_slices_partitioned = tuple(partition_all(100000, out_slices))
        for idx, part in enumerate(out_slices_partitioned):
            print(f'{idx + 1} / {len(out_slices_partitioned)}')
            start = time.time()
            fut = client.map(lambda v: renormalize_and_save_slice(src_arr, dest_arr, v, global_min, global_max, out_data_type), part)
            print(f'Submitted {len(part)} tasks to the scheduler in {time.time()- start}s')
            # wait for all the futures to complete
            result = wait(fut)
            print(f'Completed {len(part)} tasks in {time.time() - start}s')


@click.command()
@click.option('--src', '-s', type=click.Path(exists = True), help='Input zarr multiscale group/array location.')
@click.option("--dest", '-d', default='',  type=click.Path(), help='Output zarr store where shifted and masked arrays will be stored')
@click.option("--global_min", '-gmin', type=click.INT, help='Global minimum value of the input array')
@click.option("--global_max", '-gmax', type=click.INT, help='Global maximum value of the input array')
@click.option("--workers", '-w', default=100, type=click.INT, help='Number of dask scheduler workers')
@click.option('--scheduler', '-s', default="lsf", type=click.STRING)
@click.option('--data_type', '-dt', default='', type=click.STRING)
def cli(src, dest, global_min, global_max, workers, scheduler, data_type):
    
    
    if dest=='':
        dest=os.getcwd()
    
    z_store, z_path= separate_store_path(src, '')
    src_store = zarr.NestedDirectoryStore(z_store)
    source_obj = zarr.open(store=src_store,path=z_path, mode = 'r')
    
    
    dest_store = zarr.NestedDirectoryStore(dest)
    dest_root = zarr.open_group(store=dest_store, mode= 'a')

    
    if scheduler == "lsf":
        num_cores = 1
        cluster = LSFCluster(
            cores=num_cores,
            processes=num_cores,
            memory=f"{15 * num_cores}GB",
            ncpus=num_cores,
            mem=15 * num_cores,
            walltime="48:00",
            local_directory = "/scratch/$USER/"
            )
    elif scheduler == "local":
            cluster = LocalCluster()
    client = Client(cluster)
    with open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w") as text_file:
        text_file.write(str(client.dashboard_link))
    print(client.dashboard_link)

    copy_arrays(z_src=source_obj,
                dest_root=dest_root,
                global_min=global_min,
                global_max=global_max,
                client=client,
                num_workers=workers,
                comp=Zstd(level=6),
                out_data_type=data_type)


if __name__ == '__main__':
    cli()
    