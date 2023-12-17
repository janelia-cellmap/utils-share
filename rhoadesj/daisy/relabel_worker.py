from glob import glob
import os
import sys
from config import *
import daisy
from funlib.segment.arrays.impl import find_components
from funlib.segment.arrays.replace_values import replace_values

import logging
import numpy as np


def relabel_worker(tmpdir):
    nodes, edges = read_cross_block_merges(tmpdir)

    components = find_components(nodes, edges)

    print(f"Num nodes: {len(nodes)}")
    print(f"Num edges: {len(edges)}")
    print(f"Num components: {len(components)}")

    client = daisy.Client()

    while True:
        print("getting block")
        with client.acquire_block() as block:
            if block is None:
                break

            print(f"Segmenting in block {block}")

            relabel_in_block(nodes, components, block)
    print("worker finished.")


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


if __name__ == "__main__":
    # get tmpdir from command line arguments
    args = sys.argv[1:]
    relabel_worker(args[0])
