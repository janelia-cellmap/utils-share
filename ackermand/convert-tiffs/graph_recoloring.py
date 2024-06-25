import numpy as np
import pickle
import os
import tifffile
import fastremap
import networkx as nx
from pathlib import Path
import itertools
from tqdm import tqdm


def get_touching_ids(file_path, output_dir):
    print(f"Reading file {file_path}")
    im = tifffile.imread(file_path)
    print(f"Reading file {file_path} complete!")

    file_name = Path(file_path).stem

    padded_im = np.pad(im, [(1,), (1,), (1,)], "constant", constant_values=0)
    x_shape, y_shape, z_shape = im.shape
    set_of_touching_ids = set()
    total_iterations = 0

    print(f"Getting touching objects...")
    shifts = range(-1, 2)
    for x_shift, y_shift, z_shift in tqdm(
        list(itertools.product(shifts, shifts, shifts))
    ):
        if not (x_shift == 0 and y_shift == 0 and z_shift == 0):
            # can probably stop after 13 iterations since after that, in this way i have it looping, it is symmetric but with negative signs?
            total_iterations += 1
            shifted_im = padded_im[
                x_shift + 1 : x_shape + x_shift + 1,
                y_shift + 1 : y_shape + y_shift + 1,
                z_shift + 1 : z_shape + z_shift + 1,
            ]
            indices = np.where(im != shifted_im)
            im_values = list(im[indices])
            shifted_im_values = list(shifted_im[indices])
            stacked = np.column_stack([im_values, shifted_im_values])
            unique_pairs = np.unique(stacked, axis=0)
            for im_value, shifted_im_value in unique_pairs:
                if im_value != 0 and shifted_im_value != 0:
                    set_of_touching_ids.add(
                        (im_value, shifted_im_value)
                        if im_value < shifted_im_value
                        else (shifted_im_value, im_value)
                    )
            # print(len(set_of_touching_ids))
    print(f"Getting touching objects complete!")

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{file_name}_set_of_touching_ids.pkl", "wb") as handle:
        pickle.dump(set_of_touching_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return im, set_of_touching_ids


def graph_recolor_segmentation(file_path, output_dir):

    im, set_of_touching_ids = get_touching_ids(file_path, output_dir)
    file_name = Path(file_path).stem
    G = nx.Graph()
    G.add_nodes_from(list(range(1, im.max() + 1)))
    G.add_edges_from(set_of_touching_ids)
    coloring = nx.coloring.equitable_color(G, num_colors=255)
    old_values = np.array(list(coloring.keys()), dtype=np.uint16)
    new_values = np.array(list(coloring.values()), dtype=np.uint16) + 1
    output_im = im.copy()
    if output_im.dtype == np.uint8:
        output_im = output_im.astype(np.uint16)
    fastremap.remap(
        output_im,
        dict(zip(old_values, new_values)),
        preserve_missing_labels=True,
        in_place=True,
    )
    output_im = output_im.astype(np.uint8)
    os.makedirs(output_dir, exist_ok=True)
    tifffile.imwrite(
        f"{output_dir}/{file_name}_graph_relabeled.tif",
        output_im,
    )
    print(f"Done! Wrote file to {output_dir}/{file_name}_graph_relabeled.tif!")


if __name__ == "__main__":
    # get command line arguments
    import argparse

    # parse argument 1 as file path and argument 2 as output dir
    parser = argparse.ArgumentParser(description="Graph recolor segmentation")
    parser.add_argument(
        "-i", "--image_path", help="input image path", type=str, required=True
    )
    parser.add_argument(
        "-o", "--output_dir", help="output directory", type=str, required=True
    )
    args = parser.parse_args()
    graph_recolor_segmentation(args.image_path, args.output_dir)
