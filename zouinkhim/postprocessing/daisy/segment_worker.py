import daisy
import logging
import time


import skimage.measure
import skimage.filters
import skimage.morphology
from skimage.segmentation import watershed
from skimage.morphology import dilation, cube, ball
from scipy import ndimage as ndi
import daisy
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def segment_process(
    array_in, 
	roi,
    threshold=0.5,
    gaussian_kernel=2
):
    mito_dist = array_in.to_ndarray(roi, fill_value=0)
    mito_dist = skimage.filters.gaussian(mito_dist, sigma=gaussian_kernel)
    print("done gaussian", mito_dist.shape, mito_dist.max(), mito_dist.min())
    binary_peroxi = mito_dist > threshold
    print("done threshold", binary_peroxi.shape)
    markers, _ = ndi.label(binary_peroxi)
    # Apply Watershed
    ws_labels = watershed(-mito_dist, markers, mask=binary_peroxi)
    print("done watershed", ws_labels.shape)
    instance_peroxi = skimage.measure.label(ws_labels).astype(np.int64)
    print("done instance", instance_peroxi.shape, instance_peroxi.max(), instance_peroxi.min())
    instance_peroxi[binary_peroxi == 0] = 0
    print("done mask instance", instance_peroxi.shape)
    instance_peroxi = dilation(instance_peroxi, cube(2))
    #  dilation(LD[:], cube(2))

    return instance_peroxi.astype(np.uint64)


def segment_worker():

    client = daisy.Client()

    while True:
        logger.info("getting block")
        with client.acquire_block() as block:

            if block is None:
                break

            logger.info(f"got block {block}")

            
            

            logger.info(f"releasing block: {block}")


if __name__ == "__main__":

    segment_worker()