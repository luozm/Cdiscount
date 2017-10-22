"""
From training bson file split a validation file
"""

import os
from tqdm import *
import bson
import pandas as pd

import utils.utils as utils


# Input data files are available in the "../input/" directory.
data_dir = utils.data_dir
utils_dir = utils.utils_dir
train_bson_path = os.path.join(data_dir, "train.bson")

# First load the lookup tables from the CSV files.
train_offsets_df = pd.read_csv(utils_dir+"train_offsets.csv", index_col=0)
val_images_df = pd.read_csv(utils_dir+"val_images.csv", index_col=0)

num_classes = utils.num_classes
num_val_images = len(val_images_df)
train_bson_file = open(train_bson_path, "rb")
val_full_dataset = []

with tqdm(total=num_val_images) as pbar:
    for c, d in enumerate(val_images_df.itertuples()):
        offset_row = train_offsets_df.loc[d[1]]

        # Read this product's data from the BSON file.
        train_bson_file.seek(offset_row["offset"])
        item_data = train_bson_file.read(offset_row["length"])

        # Grab the image from the product.
        item = bson.BSON.decode(item_data)
        bson_img = item["imgs"][d[3]]["picture"]

        val_full_dataset.append({"x": bson_img, "y": d[2]})

        pbar.update()

val_full_dataset_df = pd.DataFrame(val_full_dataset)

val_full_dataset_df.to_pickle(utils_dir+"val_dataset.pkl")
