import argparse

import numpy as np

import os

def create_split(files, valid_prop=0.15):
    valid_count = int(valid_prop * len(files))

    valid = np.random.choice(files, valid_count, False)
    train = [file for file in files if file not in valid]
    return train, valid

def move_files(files, src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for file in files:
        src_file = os.path.join(src, file)
        dst_file = os.path.join(dst, file)
        os.symlink(src_file, dst_file)


def split(data_dir, target_dir):
    data_dir = os.path.abspath(data_dir)
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    all_files = [file for file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, file))]
    train, valid = create_split(all_files)
    move_files(train, data_dir, target_dir + "/train/")
    move_files(valid, data_dir, target_dir + "/valid/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    parser.add_argument('--target_dir', required=True,
                        help='target directory')
    args = parser.parse_args()

    split(args.data_dir, args.target_dir)