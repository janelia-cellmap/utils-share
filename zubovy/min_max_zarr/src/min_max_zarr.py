import json
import os
from typing import Literal, Tuple, Union
import dask.array as da
import zarr
import cluster_wrapper as cw
import time
import numpy as np
from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import Client, wait
from numcodecs import Zstd
from toolz import partition_all
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster

import numpy as np

import click

def separate_store_path(store: str,
                        path: str):
    new_store, path_prefix = os.path.split(store)
    if ".zarr" in path_prefix or ".n5" in path_prefix:
        return store, path
    return separate_store_path(new_store, os.path.join(path_prefix, path))

def zarr_slice_min_max(
        source: zarr.Array, 
        slices: Tuple[slice, ...],
        ):
    
    source_data = source[slices]
    #d_arr = da.from_array(source_data)
    return [np.min(source_data), np.max(source_data)] 

def get_mins_maxs(source_arr: zarr.core.Array,
                      client: Client,
                      num_workers: int,
                      dest: str):
   
    client.cluster.scale(num_workers)
    
    slices_min = []
    slices_max = []
    slices = slices_from_chunks(normalize_chunks(source_arr.chunks, shape=source_arr.shape))
    # break the slices up into batches, to make things easier for the dask scheduler
    slices_partitioned = tuple(partition_all(100000, slices))
    for idx, part in enumerate(slices_partitioned):
        print(f'{idx + 1} / {len(slices_partitioned)}')
        
        start = time.time()
        fut = client.map(lambda v: zarr_slice_min_max(source_arr, v), part)
        
        # wait for all the futures to complete
        result_min = wait(fut)
        print(f'Completed {len(part)} tasks in {time.time() - start}s')
        results= client.gather(fut)
        
        unzip_min_max = list(zip(*results))
        mins = unzip_min_max[0]
        maxs = unzip_min_max[1]
        
        slices_min.extend(mins)
        slices_max.extend(maxs)

        
    with open(os.path.join(dest, "slices_min" + ".txt"), "w") as text_file:
        text_file.write(str(slices_min))
        
    with open(os.path.join(dest, "slices_max" + ".txt"), "w") as text_file:
        text_file.write(str(slices_max))
    

@click.command()
@click.option('--src', '-s', type=click.Path(exists = True), help='Input zarr multiscale group location.')
@click.option("--dest", '-d', default='',  type=click.Path(), help='Output folder where aggregated maximums and minimums .txt files would be stored')
@click.option("--workers", '-w', default=100, type=click.INT, help='Number of dask scheduler workers')
@click.option('--scheduler', '-s', default = "lsf", type=click.STRING)
def cli(src, dest, workers, scheduler):
    
    if dest=='':
        dest=os.getcwd()
    
    z_store, z_path= separate_store_path(src, '')
    src_store = zarr.NestedDirectoryStore(z_store)
    source_arr = zarr.open_array(store=src_store,path=z_path, mode = 'r')
    
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

    get_mins_maxs(source_arr, client=client, num_workers=workers, dest=dest)

if __name__ == '__main__':
    cli()
 
