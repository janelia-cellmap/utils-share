# %%
from funlib.persistence import open_ds
from funlib.geometry import Roi, Coordinate
import numpy as np
from tifffile import imwrite
import os

# import click


# @click.group()
# @click.option(
#     "--log-level",
#     type=click.Choice(
#         ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
#     ),
#     default="INFO",
# )
# def cli(log_level):
#     logging.basicConfig(level=getattr(logging, log_level.upper()))


# @cli.command()
# @click.argument('zarr_path', type=click.Path(exists=True))
# @click.argument('zarr_dataset', type=str)
# @click.argument('tif_path', type=click.Path())
# @click.option('--invert', is_flag=True)
def zarr_to_tif(zarr_path, zarr_dataset, tif_path, invert=False, roi=None):
    print(f"Loading {zarr_path}:{zarr_dataset} ...")
    dataset = open_ds(zarr_path, zarr_dataset, mode="r")
    if roi is not None:
        img = dataset.to_ndarray(roi)
    else:
        img = dataset.to_ndarray()

    if invert:
        print("Inverting ...")
        if dataset.dtype == np.uint8:
            img = 255 - img
        else:  # assume float
            img = 1 - img

    print(f"Saving {tif_path} ...")
    os.makedirs(os.path.dirname(tif_path), exist_ok=True)
    imwrite(tif_path, img)


dataset = "jrc_mus-liver-zon-1"
# dataset = "jrc_mus-liver"
# dataset = "jrc_mus-liver-3"
level = "s5"
invert = True
roi = Roi([200000, 100000, 10000], [10000, 10000, 10000])
roi = roi.snap_to_grid((256,) * 3, mode="grow")
grow_by = Coordinate((1024,) * 3) * 4
roi = roi.grow(grow_by, grow_by)
# roi = None

zarr_path = "/nrs/cellmap/data/{dataset}/{dataset}.n5".format(dataset=dataset)
zarr_dataset = "em/fibsem-uint8/{level}".format(level=level)
# zarr_dataset = "volumes/raw/{level}".format(level=level)
tif_path = "/nrs/cellmap/rhoadesj/tmp_data/tiffs/{dataset}/{dataset}_{level}{suffix}.tif".format(
    dataset=dataset, level=level, suffix="_crop" if roi is not None else ""
)

zarr_to_tif(zarr_path, zarr_dataset, tif_path, invert=invert, roi=roi)


# if __name__ == "__main__":
#     cli()
# %%
