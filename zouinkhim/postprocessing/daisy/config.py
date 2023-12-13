from funlib.persistence import open_ds

input_file = '/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5'
# output_file = '/groups/cosem/cosem/ackermand/Cryo_FS80_Cell2_4x4x4nm_setup03_it1100000.n5'
output_file = '/nrs/cellmap/zouinkhim/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1_postprocessed.n5'
dataset = 'mito/mito'
out_dataset = 'mito/postprocessed_mito'
array_in = open_ds(input_file, dataset)
voxel_size = array_in.voxel_size
context_nm = (50*voxel_size[0],50*voxel_size[1],50*voxel_size[2]) #50 pixel overlap
chunks = array_in.data.chunks
block_size_nm = [chunks[0]*voxel_size[0],  chunks[1]*voxel_size[1], chunks[2]*voxel_size[2]]
threshold = 0.58
# min_size = 2e7 / (8**3)
gaussian_kernel = 2