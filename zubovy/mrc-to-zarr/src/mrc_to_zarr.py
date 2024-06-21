import cluster_wrapper as cw
import mrcfile
import dask.array as da
from numcodecs import Zstd

from dask.utils import SerializableLock

from fibsem_tools.io.mrc import to_dask, infer_dtype
import zarr
import time

comp = Zstd(level=6)
src_path = ''
dest_path = ''

def copy_mrc(src_path, dest_path, comp):
    mrc_file = mrcfile.open(src_path,
                            mode='r')
    
    dask_arr = to_dask(mrc_file, (1, -1, -1)).rechunk((64, -1, -1)).astype('uint16')
    dataset = zarr.create(store=zarr.NestedDirectoryStore(dest_path),
                           path='recon-1/em/fibsem-int16/s0', 
                             shape = dask_arr.shape,
                               chunks=(64,64,64),
                                 dtype=dask_arr.dtype,
                                 compressor = comp  
                                 )
    start_time = time.time()
    #lock = SerializableLock()

    da.store(dask_arr, dataset, lock = False)
    copy_time = time.time() - start_time
    print(f"({copy_time}s) copied {src_path} to {dest_path}")

if __name__ == '__main__':

    store_mrc = cw.cluster_compute("lsf", 11)(copy_mrc)
    store_mrc(src_path, dest_path, comp)

