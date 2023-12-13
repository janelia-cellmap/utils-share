from glob import glob
import os
from config import *
import daisy
from funlib.segment.arrays.impl import find_components
from funlib.segment.arrays.replace_values import replace_values

import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def relabel_worker(tmpdir):
    nodes, edges = read_cross_block_merges(tmpdir)

    components = find_components(nodes, edges)

    logger.debug("Num nodes: %s", len(nodes))
    logger.debug("Num edges: %s", len(edges))
    logger.debug("Num components: %s", len(components))

    # write_roi = daisy.Roi((0,) * len(write_size), write_size)
    read_roi = write_roi
    # total_roi = array_in.roi
    total_roi = total_roi.grow(-context, -context)

    client = daisy.Client()

    while True:
        logger.info("getting block")
        with client.acquire_block() as block:
            if block is None:
                break

            logger.debug("Segmenting in block %s", block)

            relabel_in_block(nodes, components, block)


def relabel_in_block(old_values, new_values, block):
    a = array_out.to_ndarray(block.write_roi)
    replace_values(a, old_values, new_values, inplace=True)
    array_out[block.write_roi] = a


def read_cross_block_merges(tmpdir):
    block_files = glob(os.path.join(tmpdir, "block_*.npz"))

    nodes = []
    edges = []
    for block_file in block_files:
        b = np.load(block_file)
        nodes.append(b["nodes"])
        edges.append(b["edges"])

    return np.concatenate(nodes), np.concatenate(edges)
