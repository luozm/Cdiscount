"""
Use pre-trained CNN to extract bottleneck features (last several layer before logits).
And save for fine-tuning last several layers.

Zhimeng Luo
"""

import os
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator

from utils import utils
from model.xception import extractor_3outputs

from utils.iterator import PickleIterator
from utils.iterator import BSONIterator


# ---------------------------------------------------------------------------------
# Initialization
#

# Input data files are available in the "../data/input/" directory.
data_dir = utils.data_dir
utils_dir = utils.utils_dir
train_bson_path = os.path.join(data_dir, "train.bson")

# First load the lookup tables from the CSV files.
train_product_table = pd.read_csv(utils_dir + "train_offsets.csv", index_col=0)
train_image_table = pd.read_csv(utils_dir + "train_images.csv", index_col=0)
pickle_file = pd.read_pickle(utils_dir + "val_dataset.pkl")

num_train_images = len(train_image_table)
num_val_images = len(pickle_file)
train_bson_file = open(train_bson_path, "rb")

use_crop = utils.use_crop
num_gpus = 8
batch_size = 256*num_gpus

# Create a generator for training and a generator for validation.
train_datagen = ImageDataGenerator(samplewise_center=True, rescale=1./255)
train_gen = BSONIterator(train_bson_file,
                         train_image_table,
                         train_product_table,
                         None,
                         train_datagen,
                         batch_size=batch_size,
                         labelled=False,
                         use_crop=use_crop,
                         )

val_datagen = ImageDataGenerator(samplewise_center=True, rescale=1./255)
val_gen = PickleIterator(None,
                         pickle_file,
                         val_datagen,
                         batch_size=batch_size,
                         labelled=False,
                         use_crop=use_crop,
                         )


# ---------------------------------------------------------------------------------
# Inference on multi-GPU
#

with tf.device("/cpu:0"):
    model = extractor_3outputs(95, 115, 131)

# make the model parallel
parallel_model = multi_gpu_model(model, gpus=num_gpus)

# model.summary()
bottleneck_features_train = parallel_model.predict_generator(
    train_gen,
    steps=(num_train_images // batch_size)+1,
    workers=4,
    verbose=1
)

np.save(utils_dir+'bottleneck_features_train_level1_160.npy', bottleneck_features_train[0])
np.save(utils_dir+'bottleneck_features_train_level2_160.npy', bottleneck_features_train[1])
np.save(utils_dir+'bottleneck_features_train_level3_160.npy', bottleneck_features_train[2])
print("Successfully save bottleneck_features_train")

bottleneck_features_val = parallel_model.predict_generator(
    val_gen,
    steps=(num_val_images // batch_size)+1,
    workers=4,
    verbose=1
)

np.save(utils_dir+'bottleneck_features_val_level1_160.npy', bottleneck_features_val[0])
np.save(utils_dir+'bottleneck_features_val_level2_160.npy', bottleneck_features_val[1])
np.save(utils_dir+'bottleneck_features_val_level3_160.npy', bottleneck_features_val[2])
print("Successfully save bottleneck_features_val")
