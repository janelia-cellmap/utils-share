from .impl import find_components
from .replace_values import replace_values

from funlib.persistence import Array
import daisy
import glob
import logging
import numpy as np
import os
import tempfile

logger = logging.getLogger(__name__)


def segment_blockwise(
    array_in, array_out, block_size, context, num_workers, segment_function
):
    """Segment an array in parallel.

    Args:

        array_in (``Array``):

            The input data needed by `segment_function` to produce a
            segmentation.

        array_out (``Array``):

            The array to write to. Should initially be empty (i.e., all zeros).

        block_size (``daisy.Coordinate``):

            The size of the blocks to segment (without context), in world
            units.

        context (``daisy.Coordinate``):

            The amount of padding to add (in the negative and postive
            direction) for context, in world units.

        num_workers (``int``):

            The number of workers to use.

        segment_function (function):

            A function taking arguments ``array_in`` and ``roi`` to produce a
            segmentation for ``roi`` only. Expected to return an ndarray of the
            shape of ``roi`` (using the voxel size of ``array_in``) with
            datatype ``np.uint64``. Zero in the segmentation are considered
            background and will stay zero.
    """

    write_size = daisy.Coordinate(block_size)
    write_roi = daisy.Roi((0,) * len(write_size), write_size)
    read_roi = write_roi.grow(context, context)
    total_roi = array_in.roi.grow(context, context)

    num_voxels_in_block = (read_roi / array_in.voxel_size).size

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"total_roi: {total_roi}:")
        print(f"read_roi: {read_roi}:")
        print(f"write_roi: {write_roi}:")

        task = daisy.Task(
            "segment_blockwise",
            total_roi,
            read_roi,
            write_roi,
            process_function=lambda b: segment_in_block(
                array_in, array_out, num_voxels_in_block, b, tmpdir, segment_function
            ),
            num_workers=num_workers,
            fit="shrink",
            read_write_conflict=True,
        )
        daisy.run_blockwise([task])

        nodes, edges = read_cross_block_merges(tmpdir)

    components = find_components(nodes, edges)

    logger.debug("Num nodes: %s", len(nodes))
    logger.debug("Num edges: %s", len(edges))
    logger.debug("Num components: %s", len(components))

    write_roi = daisy.Roi((0,) * len(write_size), write_size)
    read_roi = write_roi
    total_roi = array_in.roi

    task = daisy.Task(
        "relabel_blockwise",
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda b: relabel_in_block(array_out, nodes, components, b),
        num_workers=num_workers,
        fit="shrink",
    )

    daisy.run_blockwise([task])


def segment_in_block(
    array_in, array_out, num_voxels_in_block, block, tmpdir, segment_function
):
    logger.debug("Segmenting in block %s", block)

    segmentation = segment_function(array_in, block.read_roi)

    print("========= block %d ====== " % block.block_id[1])
    print(segmentation)

    assert segmentation.dtype == np.uint64

    id_bump = block.block_id[1] * num_voxels_in_block
    segmentation += id_bump
    segmentation[segmentation == id_bump] = 0

    logger.debug("Bumping segmentation IDs by %d", id_bump)

    # wrap segmentation into daisy array
    segmentation = Array(
        segmentation, roi=block.read_roi, voxel_size=array_in.voxel_size
    )

    # store segmentation in out array
    array_out[block.write_roi] = segmentation[block.write_roi]

    neighbor_roi = block.write_roi.grow(array_in.voxel_size, array_in.voxel_size)

    # clip segmentation to 1-voxel context
    segmentation = segmentation.to_ndarray(roi=neighbor_roi, fill_value=0)
    neighbors = array_out.to_ndarray(roi=neighbor_roi, fill_value=0)

    unique_pairs = []

    for d in range(3):
        slices_neg = tuple(slice(None) if dd != d else slice(0, 1) for dd in range(3))
        slices_pos = tuple(
            slice(None) if dd != d else slice(-1, None) for dd in range(3)
        )

        pairs_neg = np.array(
            [segmentation[slices_neg].flatten(), neighbors[slices_neg].flatten()]
        )
        pairs_neg = pairs_neg.transpose()

        pairs_pos = np.array(
            [segmentation[slices_pos].flatten(), neighbors[slices_pos].flatten()]
        )
        pairs_pos = pairs_pos.transpose()

        unique_pairs.append(np.unique(np.concatenate([pairs_neg, pairs_pos]), axis=0))

    unique_pairs = np.concatenate(unique_pairs)
    zero_u = unique_pairs[:, 0] == 0
    zero_v = unique_pairs[:, 1] == 0
    non_zero_filter = np.logical_not(np.logical_or(zero_u, zero_v))

    logger.debug("Matching pairs with neighbors: %s", unique_pairs)

    edges = unique_pairs[non_zero_filter]
    nodes = np.unique(edges)

    logger.debug("Final edges: %s", edges)
    logger.debug("Final nodes: %s", nodes)

    np.savez_compressed(
        os.path.join(tmpdir, "block_%d.npz" % block.block_id[1]),
        nodes=nodes,
        edges=edges,
    )


def relabel_in_block(array, old_values, new_values, block):
    a = array.to_ndarray(block.write_roi)
    replace_values(a, old_values, new_values, inplace=True)
    array[block.write_roi] = a


def read_cross_block_merges(tmpdir):
    block_files = glob.glob(os.path.join(tmpdir, "block_*.npz"))

    nodes = []
    edges = []
    for block_file in block_files:
        b = np.load(block_file)
        nodes.append(b["nodes"])
        edges.append(b["edges"])

    return np.concatenate(nodes), np.concatenate(edges)