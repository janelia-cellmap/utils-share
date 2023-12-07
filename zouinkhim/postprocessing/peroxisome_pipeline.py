import skimage.measure
import skimage.filters
import skimage.morphology
import numpy as np
from skimage.segmentation import watershed
import zarr
from scipy.ndimage import binary_fill_holes
from skimage.morphology import dilation, cube

def get_clean_instance_mito(
    peroxi_dist,
    LD,
    mito,
    cell,
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
    print("done threshold", binary_peroxi.shape, binary_peroxi.max(), binary_peroxi.min())
    # get instance labels

    # fill the wholes in the binary mask
    binary_peroxi = binary_fill_holes(binary_peroxi)
    print("done filling holes", binary_peroxi.shape, binary_peroxi.max(), binary_peroxi.min())

    # watershed
    peroxi_dist = skimage.filters.gaussian(peroxi_dist, sigma = gaussian_kernel)
    markers = skimage.measure.label(binary_peroxi)
    ws_labels = watershed(-peroxi_dist, markers, mask=binary_peroxi)
    print("done watershed", ws_labels.shape, ws_labels.max(), ws_labels.min())

    # fill the wholes
    # filled_labels = np.array([binary_fill_holes(ws_labels == i) for i in np.unique(ws_labels)])
    # filled_labels = filled_labels.max(0) * ws_labels
    # print("done filling holes", filled_labels.shape, filled_labels.max(), filled_labels.min())


    instance_peroxi = skimage.measure.label(ws_labels).astype(np.int64)
    print("done instance", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())
    # relabel background to 0
    instance_peroxi[binary_peroxi == 0] = 0
    print("done mask instance", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())
    # make mask of unwanted object class overlaps

    # all_mask = LD[:] | mito[:] | (cell[:]==1) == 0
    structure_element = cube(3)
    all_mask = dilation(cell[:], structure_element) == 0
    print("done cell mask", np.unique(all_mask, return_counts=True))
    all_mask[dilation(mito[:], cube(2)) > 0] = 1

    print("done cell mask", np.unique(all_mask, return_counts=True))
    all_mask[dilation(LD[:], cube(1)) > 0] = 1

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
    return instance_peroxi.astype(np.uint64)





threshold = 0.5
min_size = 7e6 /(8**3)
gaussian_kernel = 2

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

peroxiso_data = zarr.open(
    "/nrs/cellmap/zouinkhim/predictions/v23/v22_peroxisome_funetuning_best_v20_normal_5e5_finetuned_distances_8nm_peroxisome_jrc_mus-livers_peroxisome_8nm_upsample-unet_default_one_label_finetuning_2_1_60000.n5",
    mode="r",
)["v22_funetuning_1_60000/proxisome"]


mito_data = zarr.open(
    "/nrs/cellmap/zouinkhim/predictions/v21/2023_12_06_post_processed_2.n5",
    mode="r",
)["mito_steps/Final_relabel"]


# mito_data = zarr.open(
#     "/nrs/cellmap/zouinkhim/predictions/v21/v21_mito_attention_finetuned_distances_8nm_mito_jrc_mus-livers_mito_8nm_attention-upsample-unet_default_one_label_1_345000.n5",
#     mode="r",
# )["v21_attention_1_345000/mito"]

ld_data = zarr.open(
    "/nrs/cellmap/zouinkhim/predictions/v23/ld_crop.n5", mode="r"
)["crop3"]

cell_data = zarr.open(
    "/nrs/cellmap/zouinkhim/predictions/v23/cell_crop.n5", mode="r"
)["crop2"]

output_data = "/nrs/cellmap/zouinkhim/predictions/v21/2023_12_06_post_processed_2.n5"


result = get_clean_instance_mito(
    peroxiso_data,
    mito_data,
    ld_data,
    cell_data,
    threshold=threshold,
    min_size=min_size,
    gaussian_kernel=gaussian_kernel,
)

dataset_out = "step_4"

with zarr.open(output_data) as z_out:
    if "peroxi" not in z_out:
        d = z_out.create_group("peroxi")
    if "peroxi/"+dataset_out in z_out:
        del z_out["peroxi/"+dataset_out]
    z_out.create_dataset("peroxi/"+dataset_out, data=result, dtype=np.uint64)
    for k, v in mito_data.attrs.items():
        z_out["peroxi/"+dataset_out].attrs[k] = v
