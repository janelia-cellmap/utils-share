import os
import sys
from config import *
import daisy
from funlib.persistence import Array, open_ds

import numpy as np


def segment_worker(tmpdir):
    client = daisy.Client()
    array_in = open_ds(input_file, dataset)
    array_out = open_ds(output_file, out_dataset, mode="a")

    while True:
        print("getting block")
        with client.acquire_block() as block:
            if block is None:
                break

            print("Segmenting in block %s", block)

            segmentation = segment_function(array_in,block.read_roi)

            print("========= block %d ====== " % block.block_id[1])
            print(segmentation)

            # assert segmentation.dtype == np.uint64

            # id_bump = block.block_id[1] * num_voxels_in_block
            # segmentation += id_bump
            # segmentation[segmentation == id_bump] = 0

            # print("Bumping segmentation IDs by %d", id_bump)

            # wrap segmentation into daisy array
            segmentation = Array(
                segmentation, roi=block.read_roi, voxel_size=array_in.voxel_size
            )

            # store segmentation in out array
            array_out[block.write_roi] = segmentation[block.write_roi]

            # neighbor_roi = block.write_roi.grow(
            #     array_in.voxel_size, array_in.voxel_size
            # )

            # # clip segmentation to 1-voxel context
            # segmentation = segmentation.to_ndarray(roi=neighbor_roi, fill_value=0)
            # neighbors = array_out.to_ndarray(roi=neighbor_roi, fill_value=0)

            # unique_pairs = []

            # for d in range(3):
            #     slices_neg = tuple(
            #         slice(None) if dd != d else slice(0, 1) for dd in range(3)
            #     )
            #     slices_pos = tuple(
            #         slice(None) if dd != d else slice(-1, None) for dd in range(3)
            #     )

            #     pairs_neg = np.array(
            #         [
            #             segmentation[slices_neg].flatten(),
            #             neighbors[slices_neg].flatten(),
            #         ]
            #     )
            #     pairs_neg = pairs_neg.transpose()

            #     pairs_pos = np.array(
            #         [
            #             segmentation[slices_pos].flatten(),
            #             neighbors[slices_pos].flatten(),
            #         ]
            #     )
            #     pairs_pos = pairs_pos.transpose()

            #     unique_pairs.append(
            #         np.unique(np.concatenate([pairs_neg, pairs_pos]), axis=0)
            #     )

            # unique_pairs = np.concatenate(unique_pairs)
            # zero_u = unique_pairs[:, 0] == 0
            # zero_v = unique_pairs[:, 1] == 0
            # non_zero_filter = np.logical_not(np.logical_or(zero_u, zero_v))

            # print("Matching pairs with neighbors: %s", unique_pairs)

            # edges = unique_pairs[non_zero_filter]
            # nodes = np.unique(edges)

            # print("Final edges: %s", edges)
            # print("Final nodes: %s", nodes)

            # np.savez_compressed(
            #     os.path.join(tmpdir, "block_%d.npz" % block.block_id[1]),
            #     nodes=nodes,
            #     edges=edges,
            # )

            print(f"releasing block: {block}")

    print("worker finished.")


if __name__ == "__main__":
    # get tmpdir from command line arguments
    args = sys.argv[1:]
    segment_worker(args[0])
