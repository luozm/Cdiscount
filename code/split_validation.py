"""
From training bson file split a validation file

"""

import os
from tqdm import *
import bson
import pandas as pd

from utils import utils


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
    for image in val_images_df.itertuples():
        offset_row = train_offsets_df.loc[image[1]]

        # Read this product's data from the BSON file.
        train_bson_file.seek(offset_row["offset"])
        item_data = train_bson_file.read(offset_row["length"])

        # Grab the image from the product.
        item = bson.BSON.decode(item_data)
        bson_img = item["imgs"][image[5]]["picture"]

        val_full_dataset.append({"image": bson_img,
                                 "product_id": image[1],
                                 "img_idx": image[5],
                                 "label": image[2],
                                 "label_level1": image[3],
                                 "label_level2": image[4]})

        pbar.update()

val_full_dataset_df = pd.DataFrame(val_full_dataset)

val_full_dataset_df.to_pickle(utils_dir+"val_dataset.pkl")
