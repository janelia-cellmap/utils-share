# %%
import argparse
from pathlib import Path
from glob import glob
from typing import Iterable, Union
import tifffile
import zarr
from numcodecs.gzip import GZip
import numpy as np
from tqdm import tqdm
from parse import parse

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# SET VARIABLES HERE ===========================================================
dataset = "asinus-1"

conversion_dict_list = [
    {
        "tif_base_path": "<path-to-the-data>/{dataset}/".format(dataset=dataset),
        "tif_filename_format": "{region}--{tile}--{channel}--{slice}.tif",
        "zarr_path": "<path-to-the-data>/{dataset}/{dataset}.zarr".format(
            dataset=dataset
        ),
        "zarr_key": "{dataset}/raw".format(dataset=dataset),
        "resolution": (4, 4, 40),  # in nanometers
        "offset": (0, 0, 0),
        "overwrite": True,
        # "dtype": np.uint8,  # or np.float32
    },
]


# SHOULDN'T NEED TO CHANGE ANYTHING BELOW HERE =================================


def tif_to_zarr(
    tif_base_path=None,
    # tif_filename_format="{region}--MERGED--{slice}.tif",
    tif_filename_format="{region}--{tile}--{channel}--{slice}.tif",
    zarr_path=None,
    zarr_key="raw",
    resolution=(4, 4, 40),
    offset=[0, 0, 0],
    xy_chunk_size=256,
    overwrite: bool = False,
    dtype=None,
    zip_level=6,
):
    # if no command line arguments are passed, use the conversion_dict_list
    if tif_base_path is None:
        for conversion_dict in tqdm(conversion_dict_list):
            tif_to_zarr(**conversion_dict)
        return

    # get all the tif files in the directory
    tif_base_path = Path(tif_base_path)
    assert tif_base_path.exists(), f"{tif_base_path} does not exist"
    blank_tif_string = tif_filename_format.format(
        **{
            name: "*"
            for name in parse(tif_filename_format, tif_filename_format).named.keys()
        }
    )
    # get all the tifs matching the formatstring
    tif_files = glob(str(Path(tif_base_path, blank_tif_string)))
    logger.debug(f"Found {len(tif_files)} tif files in {tif_base_path}")
    # determine number of slices and channels
    slice_max = 0
    channel_max = 0
    for tif_file in tif_files:
        parsed = parse(tif_filename_format, tif_file).named
        if parsed is None:
            logger.warning(f"Could not parse {tif_file}")
            continue
        if "slice" in parsed:
            slice_max = max(slice_max, int(parsed["slice"]))
        if "channel" in parsed:
            channel_max = max(channel_max, int(parsed["channel"]))

    if zarr_path is None:
        zarr_path = tif_base_path / "raw.zarr"

    zarr_path = Path(zarr_path)
    if zarr_key is None:
        zarr_key = "volume"

    logger.info(f"Converting {tif_base_path} to {zarr_path}:{zarr_key}")
    if Path(zarr_path, zarr_key).exists() and not overwrite:
        logger.warning(f"{zarr_path}:{zarr_key} exists. Skipping.")
        return
    elif Path(zarr_path, zarr_key).exists() and overwrite:
        logger.warning(f"{zarr_path}:{zarr_key} exists. Overwriting.")

    chunk_size = (channel_max, xy_chunk_size, xy_chunk_size, 1)
    shape = (channel_max, current_image.shape[0], current_image.shape[1], slice_max)
    logger.debug(f"\tResolution: {resolution}")
    logger.debug(f"\tOffset: {offset}")
    logger.debug(f"\tShape: {shape}")
    logger.debug(f"\tChunk size: {chunk_size}")
    logger.debug(f"\tCompression: GZip(level={zip_level})")
    logger.debug("\tWrite empty chunks: False")
    logger.debug(f"\tOverwrite: {overwrite}")
    logger.info(f"\tLoading {tif_base_path}")
    current_image = tifffile.imread(Path(tif_base_path, tif_files[0]))
    if dtype is None:
        dtype = current_image.dtype
    logger.debug(f"\tDatatype: {dtype}")
    store = zarr.zarrStore(zarr_path)
    zarr_root = zarr.group(store=store)
    logger.info(f"\tSaving to {zarr_path}:{zarr_key}")
    ds = zarr_root.create_dataset(
        name=zarr_key,
        # data=current_image,
        dtype=dtype,
        shape=shape,
        chunks=chunk_size,
        write_empty_chunks=False,
        compressor=GZip(level=zip_level),
        overwrite=overwrite,
    )
    logger.info("\tSaving attributes")
    ds.attrs["pixelResolution"] = {
        "dimensions": resolution,
        "unit": "nm",
    }
    ds.attrs["offset"] = offset
    logger.info("\tDone")

    # now loop over each of the tiff files and write them to the zarr
    for tif_file in tif_files:
        parsed = parse(tif_filename_format, tif_file).named
        if parsed is None:
            logger.warning(f"Could not parse {tif_file}")
            continue
        slice_index = int(parsed["slice"])
        channel_index = int(parsed["channel"])
        file = Path(tif_base_path, tif_file)
        logger.info(f"\tLoading {file}")
        current_image = tifffile.imread(file)
        ds[channel_index, :, :, slice_index] = current_image

    logger.info("\tDone")


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif_base_path", type=str, required=False, default=None)
    parser.add_argument("--tif_filename_format", type=str, required=False, default=None)
    parser.add_argument("--zarr_path", type=str, required=False, default=None)
    parser.add_argument("--zarr_key", type=str, required=False, default=None)
    parser.add_argument("--resolution", type=int, required=False, default=8)
    parser.add_argument("--offset", type=Iterable, required=False, default=[0, 0, 0])
    parser.add_argument("--chunk_size", type=int, required=False, default=128)
    parser.add_argument("--ndims", type=int, required=False, default=3)
    parser.add_argument("-o", "--overwrite", action="store_true", default=False)
    args = parser.parse_args()
    tif_to_zarr(
        # args.tif_base_path, args.zarr_path, args.zarr_key, args.resolution, args.offset
        **vars(args)
    )

# %%
