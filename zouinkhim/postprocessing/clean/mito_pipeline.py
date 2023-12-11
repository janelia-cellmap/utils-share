import skimage.measure
import skimage.filters
import skimage.morphology
import numpy as np
import zarr
from skimage.segmentation import watershed
from skimage.morphology import dilation, cube, ball
from scipy import ndimage as ndi


import click

import logging


def get_clean_instance_mito_generator(
    mito_dist,
    LD,
    cell,
    nucleus,
    threshold=0.5,
    min_size=5e6,
    gaussian_kernel=5
):
    # Gaussian smooth distance predictions
    mito_dist = skimage.filters.gaussian(mito_dist, sigma=gaussian_kernel)

    print("done gaussian", mito_dist.shape, mito_dist.max(), mito_dist.min())

    # Threshold predictions
    binary_peroxi = mito_dist > threshold
    print("done threshold", binary_peroxi.shape)


    markers, _ = ndi.label(binary_peroxi)
    # Apply Watershed
    ws_labels = watershed(-mito_dist, markers, mask=binary_peroxi)
    print("done watershed", ws_labels.shape)

    mito_dist = None
    binary_peroxi = None

    # Get instance labels
    instance_peroxi = skimage.measure.label(ws_labels).astype(np.int64)
    print("done instance", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())
    # yield "3_Instance_labeled", instance_peroxi

    ws_labels = None

    # Relabel background to 0
    instance_peroxi[binary_peroxi == 0] = 0
    print("done mask instance", instance_peroxi.shape)
    #  dilation(LD[:], cube(2))
    all_mask = (dilation((1-cell[:]), cube(10)) == 0) | (dilation(LD[:], cube(2)) > 0) | (dilation(nucleus[:], cube(2)) > 0)


    instance_peroxi[all_mask] = 0
    
    print("done mask", all_mask.shape)


    # Find ids of objects that overlap with unwanted object classes
    # bad_ids = np.unique(instance_peroxi[all_mask])

    
    all_mask = None


    # # Set bad ids to 0
    # for id in bad_ids:
    #     instance_peroxi[instance_peroxi == id] = 0
    print("done bad ids", instance_peroxi.shape)   

    # Remove small objects
    instance_peroxi = skimage.morphology.remove_small_objects(
        instance_peroxi, min_size=min_size
    )
    print("done remove small", instance_peroxi.shape)

    # Relabel objects to smallest range of integers
    instance_peroxi = skimage.measure.label(instance_peroxi)
    print("done relabel", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())

    # dilate result
    structure_element = ball(3)
    instance_peroxi = dilation(instance_peroxi, structure_element)
    print("done dilate", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())

    return instance_peroxi.astype(np.uint64)



@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.command()
@click.option("-p", "--prediction", type=str)
def process(prediction):

    threshold = 0.58

    # threshold = 127
    min_size = 2e7 / (8**3)
    gaussian_kernel = 2

#   -d cell_8_center  ld_8_center mito peroxisome -f /nrs/cellmap/ackermand/cellmap/jrc_mus-liver-zon-1.n5 -d nucleus

    # zarr_mito = zarr.open(
    #     "/nrs/cellmap/zouinkhim/predictions/post_proceesing/v21_mito_attention_finetuned_distances_8nm_mito_jrc_mus-livers_mito_8nm_attention-upsample-unet_default_one_label_1_345000.n5",
    #     mode="r",
    # )

    zarr_masks = zarr.open("/nrs/cellmap/zouinkhim/predictions/post_proceesing/for_presentation.n5", mode="r")


    # result_path = "/nrs/cellmap/zouinkhim/predictions/post_proceesing/post_processed.n5"
    result_path = "/nrs/cellmap/zouinkhim/predictions/post_proceesing/for_presentation.n5"
    
    print("processing", prediction)
    mito_data = zarr_masks["mito"]
    ld_data = zarr_masks["ld_8_"+prediction]
    cell_data = zarr_masks["cell_8_"+prediction]
    nucleus_data = zarr_masks["nucleus_8_center"]
    out_data = "mito_"+prediction


    result = get_clean_instance_mito_generator(
        mito_data,
        ld_data,
        cell_data,
        nucleus_data,
        threshold=threshold,
        min_size=min_size,
        gaussian_kernel=gaussian_kernel,
    )
    with zarr.open(result_path, mode="a") as out_zarr :
        print("writing", result_path, out_data)
        if "mito_postprocessed" not in out_zarr:
            out_zarr.create_group("mito_postprocessed")
        d = out_zarr["mito_postprocessed"]
        # if out_data in d:
        #     del d[out_data]
        
        d.create_dataset(out_data, data=result, dtype=result.dtype)
        for k, v in mito_data.attrs.items():
            d[out_data].attrs[k] = v
        
        print("done writing", result_path, out_data)

if __name__ == "__main__":
    cli()
