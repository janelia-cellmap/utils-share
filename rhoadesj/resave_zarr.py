# %%
from funlib.persistence import open_ds, prepare_ds
import numpy as np
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
def zarr_to_tif(in_path, in_name, out_path, out_name, chunk_shape=None, invert=False):
    raise NotImplementedError
    print(f"Loading {in_path}:{in_name} ...")
    in_ds = open_ds(in_path, in_name, mode="r")

    if invert:
        print("Inverting ...")
        if in_ds.dtype == np.uint8:
            img = 255 - img
        else:  # assume float
            img = 1 - img

    print(f"Saving {out_path} ...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imwrite(out_path, img)


# dataset = "jrc_mus-liver-zon-2"
# dataset = "jrc_mus-liver"
dataset = "jrc_mus-liver-3"
level = "s5"
invert = True

zarr_path = "/nrs/cellmap/data/{dataset}/{dataset}.n5".format(dataset=dataset)
zarr_dataset = "em/fibsem-uint8/{level}".format(level=level)
# zarr_dataset = "volumes/raw/{level}".format(level=level)
tif_path = (
    "/nrs/cellmap/rhoadesj/tmp_data/tiffs/{dataset}/{dataset}_{level}.tif".format(
        dataset=dataset, level=level
    )
)

zarr_to_tif(zarr_path, zarr_dataset, tif_path, invert=invert)


# if __name__ == "__main__":
#     cli()
# %%
