
# from blockwise_segmentation_function import *
import funlib.segment.arrays as fsa
import argparse
import os
import numpy as np
import multiprocessing
from funlib.persistence import open_ds, prepare_ds

import skimage.measure
import skimage.filters
import skimage.morphology
from skimage.segmentation import watershed
from skimage.morphology import dilation, cube, ball
from scipy import ndimage as ndi
import daisy


input_file = '/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5'
# output_file = '/groups/cosem/cosem/ackermand/Cryo_FS80_Cell2_4x4x4nm_setup03_it1100000.n5'
output_file = '/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1_postprocessed.n5'
dataset = 'mito/mito'
out_dataset = 'mito/postprocessed_mito'
# 
num_processors = int(multiprocessing.cpu_count()/2)




def blockwise_segmentation_function(
    array_in, 
	# array_ld, 
	# array_cell, 
	# array_nucleus,
	roi,
    threshold=0.5,
    # min_size=5e6,
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
    #  dilation(LD[:], cube(2))

    return instance_peroxi.astype(np.uint64)

if __name__ == '__main__':

	try:
		os.mkdir(output_file)
	except:
		pass

	# file1 = open(f"{output_file}/input.txt", "w") 
	# file1.write(input_file) 
	# file1.close() 
	# cell_8_center    mito  nucleus_8_center  peroxisome
	array_in = open_ds(input_file, dataset)
	# array_ld = open_ds(input_file, "ld_8_center")
	# array_cell = open_ds(input_file, "cell_8_center")
	# array_nucleus = open_ds(input_file, "nucleus_8_center")
	
    
	voxel_size = array_in.voxel_size
	context_nm = (50*voxel_size[0],50*voxel_size[1],50*voxel_size[2]) #50 pixel overlap
	chunks = array_in.data.chunks
	block_size_nm = [chunks[0]*voxel_size[0],  chunks[1]*voxel_size[1], chunks[2]*voxel_size[2]]
	threshold = 0.58
	# min_size = 2e7 / (8**3)
	gaussian_kernel = 2
	
	# client = daisy.Client()

	array_out = prepare_ds(output_file,
								out_dataset,
								array_in.roi,
								voxel_size = voxel_size,
								write_size= block_size_nm,
								dtype = np.uint64)

	fsa.segment_blockwise(array_in,
							array_out,
							block_size = block_size_nm,
							context = context_nm,
							num_workers = num_processors,
							segment_function = lambda array_in,roi: blockwise_segmentation_function(array_in, 
																			#    array_ld, 
																			#    array_cell, 
																			#    array_nucleus, 
																			   roi, 
																			   threshold))