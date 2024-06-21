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


def get_clean_instance_peroxisome(
    peroxi_dist,
    LD,
    mito,
    cell,
    nucleus,
    # epithelial,
    # hepatocyte,
    threshold=0.5,
    min_size=5e6,
    gaussian_kernel=5,
):
    # gaussian smooth distance predictions
    peroxi_dist = skimage.filters.gaussian(peroxi_dist, gaussian_kernel)
    print("done gaussian", peroxi_dist.shape, peroxi_dist.max(), peroxi_dist.min())
    # threshold precictions
    binary_peroxi = peroxi_dist > threshold
    #  fill the wholes
    # binary_peroxi = binary_fill_holes(binary_peroxi)
    print("done threshold", binary_peroxi.shape, binary_peroxi.max(), binary_peroxi.min())
    # get instance labels

    # fill the wholes in the binary mask
    # binary_peroxi = binary_fill_holes(binary_peroxi)
    # print("done filling holes", binary_peroxi.shape, binary_peroxi.max(), binary_peroxi.min())

    # watershed
    peroxi_dist = skimage.filters.gaussian(peroxi_dist, sigma = gaussian_kernel)
    markers = skimage.measure.label(binary_peroxi)
    ws_labels = watershed(-peroxi_dist, markers, mask=binary_peroxi)
    peroxi_dist = None
    print("done watershed", ws_labels.shape, ws_labels.max(), ws_labels.min())

    # fill the wholes
    # filled_labels = np.array([binary_fill_holes(ws_labels == i) for i in np.unique(ws_labels)])
    # filled_labels = filled_labels.max(0) * ws_labels
    # print("done filling holes", filled_labels.shape, filled_labels.max(), filled_labels.min())


    instance_peroxi = skimage.measure.label(ws_labels).astype(np.int64)
    print("done instance", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())
    # relabel background to 0
    instance_peroxi[binary_peroxi == 0] = 0
    binary_peroxi = None
    print("done mask instance", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())
    # make mask of unwanted object class overlaps
    #  (dilation(LD[:], cube(2)) == 0 )
 
    all_mask = (dilation((1-cell[:]), cube(10)) == 0) | (dilation(LD[:], cube(2)) > 0) | (mito[:] >127) | (dilation(nucleus[:], cube(2)) > 0)

    print("done cell mask", np.unique(all_mask, return_counts=True))
    # all_mask[dilation(LD[:], cube(1)) > 0] = 1

    
    ws_labels = None


    print("done cell mask", np.unique(all_mask, return_counts=True))
    print("done mask", all_mask.shape, all_mask.max(), all_mask.min())
    # find ids of peroxisomes that overlap with unwanted object classes
    # bad_ids = np.unique(instance_peroxi[all_mask])
    # print("bad ids", bad_ids.shape)
    # set bad ids to 0
    # for id in bad_ids:
    #     instance_peroxi[instance_peroxi == id] = 0
    # faster
    # bad_id_mask = np.isin(instance_peroxi, bad_ids)
    # print("bad id mask", bad_id_mask.shape, bad_id_mask.max(), bad_id_mask.min())

    # Set all bad ids to 0
    # instance_peroxi[all_mask] = 0
    instance_peroxi[all_mask==1] = 0
    all_mask = None
    print("done bad ids", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())
    # remove small objects
    instance_peroxi = skimage.morphology.remove_small_objects(
        instance_peroxi, min_size=min_size
    )
    print("done remove small", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min() )
    # relabel objects to smallest range of integers
    instance_peroxi = skimage.measure.label(instance_peroxi)
    print("done relabel", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())
    # print("done relabel", instance_peroxi.shape)
    structure_element = ball(4)
    instance_peroxi = dilation(instance_peroxi, structure_element)
    print("done dilate", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())

    # instance_peroxi = remove_small_holes(instance_peroxi)

    

    return instance_peroxi.astype(np.uint64)





# threshold = 0.6
# # min_size = 1e6 /(8**3)
# gaussian_kernel = 2

# ===============================================================================
# 1. Load data

# ld
# /nrs/cellmap/zouinkhim/predictions/v23/ld_crop.n5, crop3

# mito
# /nrs/cellmap/zouinkhim/predictions/v21/v21_mito_attention_finetuned_distances_8nm_mito_jrc_mus-livers_mito_8nm_attention-upsample-unet_default_one_label_1_345000.n5 v21_attention_1_345000/mito

# cell
# /nrs/cellmap/zouinkhim/predictions/v23/cell_crop.n5
# crop2

# peroxisome
# /nrs/cellmap/zouinkhim/predictions/v23/v22_peroxisome_funetuning_best_v20_normal_5e5_finetuned_distances_8nm_peroxisome_jrc_mus-livers_peroxisome_8nm_upsample-unet_default_one_label_finetuning_2_1_60000.n5 v22_funetuning_1_60000/proxisome

# peroxiso_data = zarr.open(
#     "/nrs/cellmap/zouinkhim/predictions/v23/v22_peroxisome_funetuning_best_v20_normal_5e5_finetuned_distances_8nm_peroxisome_jrc_mus-livers_peroxisome_8nm_upsample-unet_default_one_label_finetuning_2_1_60000.n5",
#     mode="r",
# )["v22_funetuning_1_60000/proxisome"]


# mito_data = zarr.open(
#     "/nrs/cellmap/zouinkhim/predictions/v21/2023_12_06_post_processed_2.n5",
#     mode="r",
# )["mito_steps/Final_relabel"]


# # mito_data = zarr.open(
# #     "/nrs/cellmap/zouinkhim/predictions/v21/v21_mito_attention_finetuned_distances_8nm_mito_jrc_mus-livers_mito_8nm_attention-upsample-unet_default_one_label_1_345000.n5",
# #     mode="r",
# # )["v21_attention_1_345000/mito"]

# ld_data = zarr.open(
#     "/nrs/cellmap/zouinkhim/predictions/v23/ld_crop.n5", mode="r"
# )["crop3"]

# cell_data = zarr.open(
#     "/nrs/cellmap/zouinkhim/predictions/v23/cell_crop.n5", mode="r"
# )["crop2"]

# output_data = "/nrs/cellmap/zouinkhim/predictions/v21/2023_12_06_post_processed_2.n5"



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

    threshold = 0.50

    # threshold = 127
    min_size = 1e6 / (8**3)
    gaussian_kernel = 2



    zarr_masks = zarr.open("/nrs/cellmap/zouinkhim/predictions/post_proceesing/for_presentation.n5", mode="r")


    # result_path = "/nrs/cellmap/zouinkhim/predictions/post_proceesing/post_processed.n5"
    result_path = "/nrs/cellmap/zouinkhim/predictions/post_proceesing/for_presentation.n5"
    
    print("processing", prediction)
    peroxiso_data = zarr_masks["peroxisome"]
    mito_data = zarr_masks["mito"]
    ld_data = zarr_masks["ld_8_"+prediction]
    cell_data = zarr_masks["cell_8_"+prediction]
    nucleus_data = zarr_masks["nucleus_8_center"]
    out_data = "peroxisome_"+prediction



    result = get_clean_instance_peroxisome(
        peroxiso_data,
        ld_data,
        mito_data,
        cell_data,
        nucleus_data,
        threshold=threshold,
        min_size=min_size,
        gaussian_kernel=gaussian_kernel,
    )



    with zarr.open(result_path) as z_out:
        print("writing to", result_path, out_data)
        if "peroxisome_postprocessed" not in z_out:
            z_out.create_group("peroxisome_postprocessed")
        d = z_out["peroxisome_postprocessed"]
        # if out_data in d:
        #     print("deleting", out_data)
        #     del d[out_data]
        d.create_dataset(out_data, data=result, dtype=np.uint64)
        for k, v in mito_data.attrs.items():
            d[out_data].attrs[k] = v

        print("done writing", result_path, out_data)

if __name__ == "__main__":
    cli()
