# %%
from glob import glob
import os
import shutil
from pathlib import Path
import argparse
import numpy as np
from PIL import Image


def main(base_folder, output_folder, dataset):
    os.makedirs(output_folder, exist_ok=True)
    seg_input_prefix = "slice_"
    raw_input_prefix = "raw_"
    folders = ["yz", "xy", "xz"]
    for folder in folders:
        input_folder = os.path.join(base_folder, folder)
        if not os.path.exists(input_folder):
            input_folder = os.path.join(base_folder, folder.upper())
        print(f"Processing {input_folder} ...")
        for seg_file in glob(os.path.join(input_folder, f"{seg_input_prefix}*.tif")):
            print(f"\tLoading {seg_file} ...")
            seg = np.array(Image.open(seg_file)).astype(np.uint32)
            print(f"\t\tSeg shape: {seg.shape}")
            raw_file = os.path.join(
                input_folder,
                seg_file.removeprefix(input_folder + os.sep).replace(
                    seg_input_prefix, raw_input_prefix
                ),
            )
            slice_num = Path(seg_file).stem.removeprefix(seg_input_prefix)
            out_raw = "undefined"
            try:
                out_raw = os.path.join(
                    output_folder,
                    f"{dataset.replace(os.sep, '_')}_{folder}_{slice_num}.tif",
                )
                raw = np.array(Image.open(raw_file))
                print(f"\t\tRaw shape: {raw.shape}")
                assert raw.shape == seg.shape
                print(f"\tResaving {raw_file} to {out_raw} ...")
                shutil.copy(raw_file, out_raw)
            except Exception as e:
                print(f"\t\tFailed to resave {raw_file} to {out_raw}\n\t{e}")
                continue
            out_seg = os.path.join(
                output_folder,
                f"{dataset.replace(os.sep, '_')}_{folder}_{slice_num}_seg.npy",
            )
            # out_seg = os.path.join(output_folder, f"{dataset}_{folder}_{slice_num}_seg.tif")
            print(f"\tSaving {out_seg}...")
            np.save(
                out_seg, np.array({"masks": seg, "outlines": [], "filename": out_raw})
            )
            # Image.fromarray(seg).save(out_seg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_folder",
        help="Base folder path, such as '/prfs/cellmap/cellmap/annotations/amira/jrc_mus-heart-1/whole_cell_single_slices/'",
    )
    parser.add_argument(
        "output_folder",
        help="Output folder path, such as '/nrs/cellmap/rhoadesj/tmp_data/whole_cell_single_slices/jrc_mus-heart-1'",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset name, such as 'jrc_mus-heart-1'",
    )
    args = parser.parse_args()

    main(args.base_folder, args.output_folder, args.dataset)
