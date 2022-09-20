import gdown
import shutil
import json
import numpy as np
import argparse


def donwload_data(
    train_set: bool,
    valid_set: bool,
    unzip: bool,
):
    # download training set
    if train_set:
        url = "https://drive.google.com/uc?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M"
        output = "WIDER_train.zip"
        gdown.download(url, output, quiet=False)
        if unzip:
            shutil.unpack_archive(output, "WIDER_train")

    # download validation set
    if valid_set:
        url = "https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q"
        output = "WIDER_val.zip"
        gdown.download(url, output, quiet=False)
        if unzip:
            shutil.unpack_archive(output, "WIDER_val")


parser = argparse.ArgumentParser()
parser.add_argument(
    "-ts", "--train_set", type=bool, help="Download the train_set", default=True
)
parser.add_argument(
    "-vs", "--val_set", type=bool, help="Download the val_set", default=True
)
parser.add_argument(
    "-z", "--unzip", type=bool, help="unzip all the downloaded files", default=True
)
args = parser.parse_args()
donwload_data(args.train_set, args.val_set, args.unzip)