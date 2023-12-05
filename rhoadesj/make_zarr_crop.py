# %%
import argparse
from pathlib import Path
from typing import Iterable, Union
from tqdm import tqdm
from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Coordinate, Roi

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


dataset = "jrc_mus-liver-zon-2"
# dataset = "jrc_mus-liver"
level = "s0"

input_path = "/nrs/cellmap/data/{dataset}/{dataset}.n5".format(dataset=dataset)
input_dataset = "em/fibsem-uint8/{level}".format(level=level)
# input_dataset = "volumes/raw/{level}".format(level=level)
output_path = "/nrs/cellmap/rhoadesj/tmp_data/{dataset}.zarr".format(dataset=dataset)
output_dataset = "volumes/mito_test/raw"
offset = [1641, 232, 1466]  # in voxels
shape = [12500, 12500, 12500]  # in voxels

conversion_dict_list = [
    {
        "input_path": input_path,
        "input_dataset": input_dataset,
        "output_path": output_path,
        "output_dataset": output_dataset,
        "overwrite": True,
        "offset": offset,
        "shape": shape,
    },
]


def make_zarr_crop(
    input_path: Path | str | None = None,
    input_dataset: str | None = None,
    output_path: Path | str | None = None,
    output_dataset: str | None = None,
    offset: Iterable = [0, 0, 0],  # in voxels
    shape: Iterable = [256, 256, 256],  # in voxels
    overwrite: bool = False,
):
    if input_path is None:
        for conversion_dict in tqdm(conversion_dict_list):
            make_zarr_crop(**conversion_dict)
        return
    logger.info(
        f"Cropping {input_path}:{input_dataset} to {output_path}:{output_dataset}"
    )
    if Path(output_path, output_dataset).exists() and not overwrite:
        logger.warning(f"{output_path}:{output_dataset} exists. Skipping.")
        return
    elif Path(output_path, output_dataset).exists() and overwrite:
        logger.warning(f"{output_path}:{output_dataset} exists. Overwriting.")
    logger.debug(f"\tOffset: {offset}")
    logger.debug(f"\tShape: {shape}")
    logger.debug(f"\tOverwrite: {overwrite}")
    logger.info(f"\tLoading {input_path}")
    input_ds = open_ds(input_path, input_dataset, mode="r")
    dtype = input_ds.dtype
    resolution = input_ds.voxel_size
    logger.debug(f"\tDatatype: {dtype}")
    logger.debug(f"\tResolution: {resolution}")
    input_roi = Roi(offset, shape) * resolution
    if (input_path[-4:] == ".n5") == (output_path[-4:] == ".n5"):
        output_roi = input_roi
    else:
        output_roi = Roi(offset[::-1], shape[::-1]) / resolution[::-1]
    logger.info(f"\tSaving to {output_path}:{output_dataset}")
    output_ds = prepare_ds(
        output_path,
        output_dataset,
        total_roi=output_roi,
        voxel_size=resolution,
        dtype=dtype,
        delete=overwrite,
    )
    logger.info("\tCopying data")
    output_ds[output_roi] = input_ds[input_roi]
    logger.info("\tDone")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=False, default=None)
    parser.add_argument("--input_dataset", type=str, required=False, default=None)
    parser.add_argument("--output_path", type=str, required=False, default=None)
    parser.add_argument("--output_dataset", type=str, required=False, default=None)
    parser.add_argument("--offset", type=Iterable, required=False, default=[0, 0, 0])
    parser.add_argument(
        "--shape", type=Iterable, required=False, default=[256, 256, 256]
    )
    parser.add_argument("-o", "--overwrite", action="store_true", default=False)
    args = parser.parse_args()
    make_zarr_crop(**vars(args))
