import skimage.measure
import skimage.filters
import skimage.morphology
import numpy as np
from skimage.segmentation import watershed
import zarr
from scipy.ndimage import binary_fill_holes
from skimage.morphology import dilation, cube
from funlib.persistence import open_ds, prepare_ds
import daisy

input_file = (
    "/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5"
)
output_file = (
    "/nrs/cellmap/rhoadesj/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5"
)
dataset = "peroxisome/peroxisome"
out_dataset = "peroxisome/instance"

context_padding = 50
threshold = 0.51  # 0.5
min_size = 7e6 / (8**3)
gaussian_kernel = 2

# =========
array_in = open_ds(input_file, dataset)
voxel_size = array_in.voxel_size
block_size = np.array(array_in.data.chunks) * np.array(voxel_size)
write_size = daisy.Coordinate(block_size)
context = daisy.Coordinate(np.array(voxel_size) * context_padding)
write_roi = daisy.Roi((0,) * len(write_size), write_size)
read_roi = write_roi.grow(context, context)
# total_roi = array_in.roi.grow(context, context)
# total_roi = array_in.roi
total_roi = daisy.Roi((172800, 38400, 44800), (32000, 32000, 32000))
# total_roi = daisy.Roi((172800, 38400, 44800), voxel_size * 512)

num_voxels_in_block = (read_roi / array_in.voxel_size).size

try:
    array_out = open_ds(output_file, out_dataset, mode="a")
except KeyError:
    array_out = prepare_ds(
        output_file,
        out_dataset,
        total_roi,
        voxel_size=voxel_size,
        write_size=write_size,
        dtype=np.uint64,
        delete=True,
    )


def segment_function(block):
    dist = array_in.to_ndarray(block.read_roi, fill_value=0)
    # gaussian smooth distance predictions
    dist = skimage.filters.gaussian(dist, gaussian_kernel)
    print("done gaussian")
    # threshold precictions
    binary = dist > threshold
    print("done threshold")
    # get instance labels

    # fill the wholes in the binary mask
    # binary = binary_fill_holes(binary)
    # print("done filling holes")

    # watershed
    dist = skimage.filters.gaussian(dist, sigma=gaussian_kernel)
    markers = skimage.measure.label(binary)
    ws_labels = watershed(-dist, markers, mask=binary)
    print("done watershed")

    # fill the wholes
    # filled_labels = np.array([binary_fill_holes(ws_labels == i) for i in np.unique(ws_labels)])
    # filled_labels = filled_labels.max(0) * ws_labels
    # print("done filling holes")

    instance = skimage.measure.label(ws_labels).astype(np.int64)
    print("done instance")
    # relabel background to 0
    instance[binary == 0] = 0
    print("done mask instance")

    return instance.astype(np.uint64)
