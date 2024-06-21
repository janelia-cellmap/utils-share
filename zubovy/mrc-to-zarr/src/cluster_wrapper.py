import os
import zarr
import dask.array as da

from dask.distributed import Client
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster


def cluster_compute(scheduler, num_workers):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if scheduler == "lsf":
                num_cores = 7
                cluster = LSFCluster( cores=num_cores,
                        processes=1,
                        memory=f"{15 * num_cores}GB",
                        ncpus=num_cores,
                        mem=15 * num_cores,
                        walltime="48:00", 
                        death_timeout = 240.0,
                        local_directory = "/scratch/zubovy/"
                        )
                cluster.scale(num_workers)
            elif scheduler == "local":
                    cluster = LocalCluster()

            with Client(cluster) as cl:
                text_file = open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w")
                text_file.write(str(cl.dashboard_link))
                text_file.close()
                cl.compute(function(*args, **kwargs), sync=True)

        return wrapper
    return decorator
    