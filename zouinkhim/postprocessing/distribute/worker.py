import daisy
import logging
import time
from funlib.persistence import open_ds, prepare_ds
import uuid
import yaml
import skimage.measure
import skimage.filters
import skimage.morphology
import numpy as np
from funlib.persistence import Array
from scipy import ndimage as ndi

from skimage.segmentation import watershed

logger = logging.getLogger(__name__)

def blockwise_segmentation_function(
    mito_dist,
    threshold=0.5,
    gaussian_kernel=2
):
    mito_dist = skimage.filters.gaussian(mito_dist, sigma=gaussian_kernel)
    logger.info("done gaussian", mito_dist.shape, mito_dist.max(), mito_dist.min())
    binary_peroxi = mito_dist > threshold
    logger.info("done threshold"+ str(binary_peroxi.shape))
    # markers, _ = ndi.label(binary_peroxi)
    # Apply Watershed
    # ws_labels = watershed(-mito_dist, markers, mask=binary_peroxi)
    # logger.info("done watershed"+ str(ws_labels.shape))
    # instance_peroxi = skimage.measure.label(ws_labels).astype(np.int64)
    # logger.info("done instance", str(instance_peroxi.shape), instance_peroxi.max(), instance_peroxi.min())
    # logger.info("done instance" + str(instance_peroxi.shape)+ " "+str(instance_peroxi.max())+" "+ str(instance_peroxi.min()))
    # instance_peroxi[binary_peroxi == 0] = 0
    # logger.info("done mask instance"+ str(instance_peroxi.shape))
    #  dilation(LD[:], cube(2))

    return binary_peroxi.astype(np.uint64)


# def blockwise_segmentation_function(
#     mito_dist,
#     threshold=0.5,
#     gaussian_kernel=2
# ):
#     mito_dist = skimage.filters.gaussian(mito_dist, sigma=gaussian_kernel)
#     logger.info("done gaussian", mito_dist.shape, mito_dist.max(), mito_dist.min())
#     binary_peroxi = mito_dist > threshold
#     logger.info("done threshold"+ str(binary_peroxi.shape))
#     markers, _ = ndi.label(binary_peroxi)
#     # Apply Watershed
#     ws_labels = watershed(-mito_dist, markers, mask=binary_peroxi)
#     logger.info("done watershed"+ str(ws_labels.shape))
#     instance_peroxi = skimage.measure.label(ws_labels).astype(np.int64)
#     logger.info("done instance", str(instance_peroxi.shape), instance_peroxi.max(), instance_peroxi.min())
#     logger.info("done instance" + str(instance_peroxi.shape)+ " "+str(instance_peroxi.max())+" "+ str(instance_peroxi.min()))
#     instance_peroxi[binary_peroxi == 0] = 0
#     logger.info("done mask instance"+ str(instance_peroxi.shape))
#     #  dilation(LD[:], cube(2))

#     return instance_peroxi.astype(np.uint64)


# Generate a random UUID for the log file name
log_file_name = f'log/logfile_{uuid.uuid4()}.log'
logging.basicConfig(level=logging.INFO, filename=log_file_name, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



def test_worker():
    config_path = "config.yaml"
    configs = yaml.load(open(config_path), Loader=yaml.FullLoader)
    input_file = configs["input_file"]
    dataset = configs["dataset"]
    output_file = configs["output_file"]
    out_dataset = configs["out_dataset"]
    
    
    array_in = open_ds(input_file, dataset)

    chunks = array_in.data.chunks
    voxel_size = array_in.voxel_size
    block_size_nm = [chunks[0]*voxel_size[0],  chunks[1]*voxel_size[1], chunks[2]*voxel_size[2]]
    logger.info("block_size_nm", block_size_nm)

    array_out = prepare_ds(output_file,
								out_dataset,
								array_in.roi,
								voxel_size = array_in.voxel_size,
								write_size= block_size_nm,
								dtype = np.uint64)

    client = daisy.Client()
    # print(client)

    while True:
        # logger.info("getting block")
        with client.acquire_block() as block:

            if block is None:
                break
            # read input_file, dataset from yaml file
           
            mito_dist = array_in.to_ndarray(block.read_roi, fill_value=0)

            segmentation = blockwise_segmentation_function(mito_dist)
            segmentation = Array(        segmentation, roi=block.read_roi, voxel_size=array_in.voxel_size    )
            
            array_out[block.write_roi] = segmentation[block.write_roi]


            # pretend to do some work
            # time.sleep(0.5)

            logger.info(f"releasing block: {block}")
            block.status = daisy.BlockStatus.SUCCESS


if __name__ == "__main__":

    test_worker()