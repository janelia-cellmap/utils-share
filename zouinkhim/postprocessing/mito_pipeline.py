import skimage.measure
import skimage.filters
import skimage.morphology
import numpy as np
import zarr
from skimage.segmentation import watershed
from skimage.morphology import dilation, cube
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt

# def get_clean_instance_mito_generator(
#     mito_dist,
#     LD,
#     cell,
#     threshold=0.5,
#     min_size=5e6,
#     gaussian_kernel=5,
#     distance_threshold=0.7
# ):
#     # Gaussian smooth distance predictions
#     mito_dist = skimage.filters.gaussian(mito_dist, sigma = gaussian_kernel)
#     print("done gaussian", mito_dist.shape,mito_dist.max(),mito_dist.min())
#     # yield "1_Gaussian_smoothed", mito_dist

#     # Threshold predictions
#     binary_peroxi = mito_dist > threshold
#     print("done threshold", binary_peroxi.shape)
#     # yield "2_Threshold_applied", binary_peroxi
#     # distance = ndi.distance_transform_edt(binary_peroxi)
#     # local_maxi = peak_local_max(distance, threshold_abs=distance_threshold)
#     # mito_dist = skimage.filters.gaussian(mito_dist, sigma = gaussian_kernel)
#     markers = skimage.measure.label(binary_peroxi)
#     print("done markers", markers.shape, markers.max(), markers.min())
#     ws_labels = watershed(-mito_dist, markers, mask=binary_peroxi)
#     print("done watershed", ws_labels.shape)
#     yield "1_Watershed_applied", ws_labels

#     # Get instance labels
#     instance_peroxi = skimage.measure.label(ws_labels).astype(np.int64)
#     print("done instance", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())
#     # yield "3_Instance_labeled", instance_peroxi

#     # Relabel background to 0
#     instance_peroxi[binary_peroxi == 0] = 0
#     print("done mask instance", instance_peroxi.shape)
#     yield "4_Masked_instance", instance_peroxi

#     # Make mask of unwanted object class overlaps
#     # cell_mask = np.logical_not(cell).astype(cell.dtype)
#     # print("done cell mask", cell_mask.shape)
#     structure_element = cube(8)
#     cell[cell>0] = 1 
#     cell = dilation(cell, structure_element)
#     all_mask = (cell == 0) | (LD[:] == 1)


#     instance_peroxi[all_mask] = 0
    
#     print("done mask", all_mask.shape)


#     # Find ids of objects that overlap with unwanted object classes
#     # bad_ids = np.unique(instance_peroxi[all_mask])


#     # # Set bad ids to 0
#     # for id in bad_ids:
#     #     instance_peroxi[instance_peroxi == id] = 0
#     print("done bad ids", instance_peroxi.shape)   
#     yield "4_Removed_bad_IDs", instance_peroxi

#     # Remove small objects
#     instance_peroxi = skimage.morphology.remove_small_objects(
#         instance_peroxi, min_size=min_size
#     )
#     print("done remove small", instance_peroxi.shape)
#     yield "5_Small_objects_removed", instance_peroxi

#     # Relabel objects to smallest range of integers
#     instance_peroxi = skimage.measure.label(instance_peroxi)
#     print("done relabel", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())
#     yield "Final_relabel", instance_peroxi

#     return instance_peroxi.astype(np.uint64)



def get_clean_instance_mito_generator(
    mito_dist,
    LD,
    cell,
    threshold=0.5,
    min_size=5e6,
    gaussian_kernel=5,
    distance_threshold=0.7
):
    # Gaussian smooth distance predictions
    mito_dist = skimage.filters.gaussian(mito_dist, sigma=gaussian_kernel)
    print("done gaussian", mito_dist.shape, mito_dist.max(), mito_dist.min())

    # Threshold predictions
    binary_peroxi = mito_dist > threshold
    print("done threshold", binary_peroxi.shape)

    # Apply distance transform
    distance = distance_transform_edt(binary_peroxi)
    print("done distance transform", distance.shape)

    # Find local maxima as markers
    markers = skimage.measure.label(distance > distance_threshold)
    print("done markers", markers.shape, markers.max(), markers.min())

    # Apply Watershed
    ws_labels = watershed(-distance, markers, mask=binary_peroxi)
    print("done watershed", ws_labels.shape)
    yield "1_Watershed_applied", ws_labels

    # Get instance labels
    instance_peroxi = skimage.measure.label(ws_labels).astype(np.int64)
    print("done instance", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())
    # yield "3_Instance_labeled", instance_peroxi

    # Relabel background to 0
    instance_peroxi[binary_peroxi == 0] = 0
    print("done mask instance", instance_peroxi.shape)
    yield "4_Masked_instance", instance_peroxi

    # Make mask of unwanted object class overlaps
    # cell_mask = np.logical_not(cell).astype(cell.dtype)
    # print("done cell mask", cell_mask.shape)
    structure_element = cube(8)
    cell = cell[:]
    cell[cell>0] = 1 
    cell = dilation(cell, structure_element)
    all_mask = (cell == 0) | (LD[:] == 1)


    instance_peroxi[all_mask] = 0
    
    print("done mask", all_mask.shape)


    # Find ids of objects that overlap with unwanted object classes
    # bad_ids = np.unique(instance_peroxi[all_mask])


    # # Set bad ids to 0
    # for id in bad_ids:
    #     instance_peroxi[instance_peroxi == id] = 0
    print("done bad ids", instance_peroxi.shape)   
    yield "4_Removed_bad_IDs", instance_peroxi

    # Remove small objects
    instance_peroxi = skimage.morphology.remove_small_objects(
        instance_peroxi, min_size=min_size
    )
    print("done remove small", instance_peroxi.shape)
    yield "5_Small_objects_removed", instance_peroxi

    # Relabel objects to smallest range of integers
    instance_peroxi = skimage.measure.label(instance_peroxi)
    print("done relabel", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())
    yield "Final_relabel", instance_peroxi

    return instance_peroxi.astype(np.uint64)

threshold = 0.55

# threshold = 127
min_size = 2e7 / (8**3)
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

mito_data = zarr.open(
    "/nrs/cellmap/zouinkhim/predictions/v21/v21_mito_attention_finetuned_distances_8nm_mito_jrc_mus-livers_mito_8nm_attention-upsample-unet_default_one_label_1_345000.n5",
    mode="r",
)["v21_attention_1_345000/mito"]

ld_data = zarr.open(
    "/nrs/cellmap/zouinkhim/predictions/v23/ld_crop.n5", mode="r"
)["crop3"]

cell_data = zarr.open(
    "/nrs/cellmap/zouinkhim/predictions/v23/cell_crop.n5", mode="r"
)["crop2"]

output_data = "/nrs/cellmap/zouinkhim/predictions/v21/2023_12_06_post_processed_2.n5"
out_data = "mito_steps_4"

with zarr.open(output_data) as z_out:
    if out_data in z_out:
        del z_out[out_data]
    d = z_out.create_group(out_data)
    for step, result in get_clean_instance_mito_generator(
        mito_data,
        ld_data,
        cell_data,
        threshold=threshold,
        min_size=min_size,
        gaussian_kernel=gaussian_kernel,
    ):
        if step in d:
            del d[step]
        
        d.create_dataset(step, data=result, dtype=np.uint64)
        for k, v in mito_data.attrs.items():
            d[step].attrs[k] = v
