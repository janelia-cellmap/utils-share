import skimage.measure
import skimage.filters
import skimage.morphology
from skimage.segmentation import watershed
from skimage.morphology import dilation, cube, ball
from scipy import ndimage as ndi
import daisy
import numpy as np
from funlib.persistence import open_ds, prepare_ds


from funlib.persistence import open_ds

input_file = (
    "/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5"
)
# output_file = '/groups/cosem/cosem/ackermand/Cryo_FS80_Cell2_4x4x4nm_setup03_it1100000.n5'
output_file = "/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1_postprocessed.n5"
dataset = "mito/mito"
out_dataset = "mito/postprocessed_mito"
array_in = open_ds(input_file, dataset)
voxel_size = array_in.voxel_size
block_size = np.array(array_in.data.chunks) * np.array(voxel_size)
write_size = daisy.Coordinate(block_size)
context = daisy.Coordinate(np.array(voxel_size) * 10)  # 50 pixel overlap
threshold = 0.58
# min_size = 2e7 / (8**3)
gaussian_kernel = 2

write_roi = daisy.Roi((0,) * len(write_size), write_size)
read_roi = write_roi.grow(context, context)
total_roi = array_in.roi.grow(context, context)

num_voxels_in_block = (read_roi / array_in.voxel_size).size

# try:
#     array_out = open_ds(output_file, out_dataset, mode="a")
# except KeyError:
#     array_out = prepare_ds(
#         output_file,
#         out_dataset,
#         total_roi,
#         voxel_size=voxel_size,
#         write_size=write_size,
#         dtype=np.uint64,
#     )


def segment_function(block):
    dist = array_in.to_ndarray(block.read_roi, fill_value=0)
    dist = skimage.filters.gaussian(dist, sigma=gaussian_kernel)
    print("done gaussian")
    binary = dist > threshold
    print("done threshold")
    markers, _ = ndi.label(binary)
    # Apply Watershed
    ws_labels = watershed(-dist, markers, mask=binary)
    print("done watershed")
    instance = skimage.measure.label(ws_labels).astype(np.int64)
    print("done instance")
    instance[binary == 0] = 0
    print("done mask instance")
    instance = dilation(instance, cube(2))
    #  dilation(LD[:], cube(2))

    return instance.astype(np.uint64)
