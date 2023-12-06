# %%
import argparse
from pathlib import Path
from typing import Iterable, Union
import tifffile
import zarr
from numcodecs.gzip import GZip
import numpy as np
from tqdm import tqdm
from funlib.persistence import open_ds

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


dataset = "jrc_mus-liver-zon-1"
# dataset = "jrc_mus-liver"
level = "s5"

zarr_path = "/nrs/cellmap/data/{dataset}/{dataset}.n5".format(dataset=dataset)
zarr_dataset = "em/fibsem-uint8/{level}".format(level=level)
# zarr_dataset = "volumes/raw/{level}".format(level=level)
tif_path = "/nrs/cellmap/rhoadesj/tmp_data/tiffs/{dataset}/{dataset}_{level}_cp_masks.tif".format(
    dataset=dataset, level=level
)
ds = open_ds(zarr_path, zarr_dataset, mode="r")

conversion_dict_list = [
    {
        "image_path": tif_path,
        "n5_path": "/nrs/cellmap/rhoadesj/tmp_data/{dataset}.n5".format(
            dataset=dataset
        ),
        "dataset_name": "volumes/predictions/cells",
        "resolution": ds.voxel_size,
        "offset": ds.roi.offset,
        "overwrite": True,
        "dtype": np.uint32,
    },
]


def tif_to_n5(
    image_path: Path | str | None = None,
    n5_path: Path | str | None = None,
    dataset_name: str | None = None,
    resolution: int | float | Iterable = 8,
    offset: Iterable = [0, 0, 0],
    chunk_size: int = 128,
    ndims: int = 3,
    overwrite: bool = False,
    dtype: np.dtype | None = None,
):
    if image_path is None:
        for conversion_dict in tqdm(conversion_dict_list):
            tif_to_n5(**conversion_dict)
        return
    image_path = Path(image_path)
    if n5_path is None:
        n5_path = image_path.parent / (image_path.name.removesuffix(".tif") + ".n5")
        dataset_name = "volume"
    n5_path = Path(n5_path)
    if dataset_name is None:
        dataset_name = image_path.name.removesuffix(".tif")
    logger.info(f"Converting {image_path} to {n5_path}:{dataset_name}")
    if Path(n5_path, dataset_name).exists() and not overwrite:
        logger.warning(f"{n5_path}:{dataset_name} exists. Skipping.")
        return
    elif Path(n5_path, dataset_name).exists() and overwrite:
        logger.warning(f"{n5_path}:{dataset_name} exists. Overwriting.")
    logger.debug(f"\tResolution: {resolution}")
    logger.debug(f"\tOffset: {offset}")
    logger.debug(f"\tChunk size: {chunk_size}")
    logger.debug(f"\tNumber of dimensions: {ndims}")
    logger.debug("\tCompression: GZip(level=6)")
    logger.debug("\tWrite empty chunks: False")
    logger.debug(f"\tOverwrite: {overwrite}")
    logger.info(f"\tLoading {image_path}")
    current_image = tifffile.imread(image_path)
    if dtype is None:
        dtype = current_image.dtype
    logger.debug(f"\tDatatype: {dtype}")
    store = zarr.N5Store(n5_path)
    zarr_root = zarr.group(store=store)
    logger.info(f"\tSaving to {n5_path}:{dataset_name}")
    ds = zarr_root.create_dataset(
        name=dataset_name,
        data=current_image,
        dtype=dtype,
        shape=current_image.shape,
        chunks=chunk_size,
        # write_empty_chunks=True,
        write_empty_chunks=False,
        compressor=GZip(level=6),
        overwrite=overwrite,
    )
    logger.info("\tSaving attributes")
    if not isinstance(resolution, Iterable):
        resolution = [resolution] * ndims
    ds.attrs["pixelResolution"] = {
        "dimensions": resolution,
        "unit": "nm",
    }
    ds.attrs["offset"] = offset
    logger.info("\tDone")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=False, default=None)
    parser.add_argument("--n5_path", type=str, required=False, default=None)
    parser.add_argument("--dataset_name", type=str, required=False, default=None)
    parser.add_argument("--resolution", type=int, required=False, default=8)
    parser.add_argument("--offset", type=Iterable, required=False, default=[0, 0, 0])
    parser.add_argument("--chunk_size", type=int, required=False, default=128)
    parser.add_argument("--ndims", type=int, required=False, default=3)
    parser.add_argument("-o", "--overwrite", action="store_true", default=False)
    args = parser.parse_args()
    tif_to_n5(
        # args.image_path, args.n5_path, args.dataset_name, args.resolution, args.offset
        **vars(args)
    )
