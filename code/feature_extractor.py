"""
Use pre-trained CNN to extract bottleneck features (last several layer before logits).
And save for fine-tuning last several layers.

Zhimeng Luo
"""

import os
import io
import numpy as np
import pandas as pd
import bson


import tensorflow as tf
import keras
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import utils.utils as utils
from model import *


# The Keras generator is implemented by the `BSONIterator` class.
# It creates batches of images (and their one-hot labels) directly from the BSON file.
# It can be used with multiple workers.
#
# See also the code in: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
class BSONIterator(Iterator):
    def __init__(self, bson_file, image_table, product_table, num_label,
                 image_data_generator, num_label_level1=None, num_label_level2=None,
                 image_size=(180, 180), labelled=True, batch_size=32,
                 shuffle=False, seed=None, use_hierarchical_label=False):
        """
        :param bson_file:               The original bson file instance
        :param image_table:             The table that stores information of each image
        :param product_table:           The table that stores position of each product
        :param num_label:               The number of categories
        :param image_data_generator:    Data augmentation generator
        :param num_label_level1:        The number of categories in level1
        :param num_label_level2:        The number of categories in level2
        :param image_size:              The size of image
        :param labelled:                Check if the image is of validation or training set
        :param batch_size:              Size of the batch
        :param shuffle:                 Whether shuffle dataset or not
        :param seed:                    Random seed
        :param use_hierarchical_label:  Whether use hierarchical label or not
        """
        # Parameter initialization
        self.__file = bson_file
        self.__image_table = image_table
        self.__product_table = product_table
        self.__labelled = labelled
        self.__num_images = len(image_table)
        self.__num_label = num_label
        self.__num_label_level1 = num_label_level1
        self.__num_label_level2 = num_label_level2
        self.__image_data_generator = image_data_generator
        self.__image_size = tuple(image_size)
        self.__image_shape = self.__image_size + (3,)
        self.__use_hierarchical_label = use_hierarchical_label
        # Pass parameter back to super class
        super(BSONIterator, self).__init__(self.__num_images, batch_size, shuffle, seed)

        # Print out information
        print("Found %d images." % self.__num_images)

    def _get_batches_of_transformed_samples(self, index_array):
        """
        :param index_array:
        :return: The batch of samples(pair of image and label)
        """
        batch_x = np.zeros((len(index_array),) + self.__image_shape, dtype=K.floatx())
        if self.__labelled:
            batch_y = np.zeros((len(batch_x), self.__num_label), dtype=K.floatx())
            if self.__use_hierarchical_label:
                batch_y_level1 = np.zeros((len(batch_x), self.__num_label_level1), dtype=K.floatx())
                batch_y_level2 = np.zeros((len(batch_x), self.__num_label_level2), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and data frame access with a lock.
            with self.lock:
                image_row = self.__image_table.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.__product_table.loc[product_id]

                # Read this product's data from the BSON file.
                self.__file.seek(offset_row["offset"])
                item_data = self.__file.read(offset_row["length"])

            # Grab the image from the product.
            item = bson.BSON.decode(item_data)
            img_idx = image_row["img_idx"]
            bson_img = item["imgs"][img_idx]["picture"]

            # Pre process the image.
            img = load_img(io.BytesIO(bson_img), target_size=self.__image_size)
            x = img_to_array(img)
            x = self.__image_data_generator.random_transform(x)
            x = self.__image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.__labelled:
                batch_y[i, image_row["category_idx"]] = 1
                if self.__use_hierarchical_label:
                    batch_y_level1[i, image_row["category_idx_level1"]] = 1
                    batch_y_level2[i, image_row["category_idx_level2"]] = 1

        if self.__labelled:
            if not self.__use_hierarchical_label:
                return batch_x, batch_y
            else:
                return batch_x, [batch_y_level1, batch_y_level2, batch_y]
        else:
            return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)


class PickleGenerator(Iterator):
    def __init__(self, num_label, pickle_file, image_data_generator, batch_size=32,
                 num_label_level1=None, num_label_level2=None, use_hierarchical_label=False,
                 image_size=(180, 180), labelled=True, shuffle=False, seed=None,):
        """
        :param pickle_file:             The original bson file instance
        :param num_label:               The amount of categories
        :param num_label_level1:        The number of categories in level1
        :param num_label_level2:        The number of categories in level2
        :param use_hierarchical_label:  Whether use hierarchical label or not
        :param image_data_generator:    Data augmentation generator
        :param image_size:              The size of image
        :param labelled:                Check if the image is of validation or training set
        :param batch_size:              Size of the batch
        :param shuffle:                 Whether use the shuffle strategy
        :param seed:                    Random seed
        """
        self.__pickle_file = pickle_file
        self.__batch_size = batch_size
        self.__labelled = labelled
        self.__num_label = num_label
        self.__num_label_level1 = num_label_level1
        self.__num_label_level2 = num_label_level2
        self.__image_data_generator = image_data_generator
        self.__image_size = tuple(image_size)
        self.__image_shape = self.__image_size + (3,)
        self.__num_images = len(pickle_file)
        self.__use_hierarchical_label = use_hierarchical_label

        # pass arguments to super class
        super(PickleGenerator, self).__init__(self.__num_images, batch_size, shuffle, seed)
        # Print out information
        print("Found %d images." % self.__num_images)

    def next(self):
        index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        """
        :param index_array:
        :return: Batch of samples(pair of image and label)
        """
        batch_x = np.zeros((len(index_array),) + self.__image_shape, dtype=K.floatx())
        if self.__labelled:
            batch_y = np.zeros((len(batch_x), self.__num_label), dtype=K.floatx())
            if self.__use_hierarchical_label:
                batch_y_level1 = np.zeros((len(batch_x), self.__num_label_level1), dtype=K.floatx())
                batch_y_level2 = np.zeros((len(batch_x), self.__num_label_level2), dtype=K.floatx())

        for i, j in enumerate(index_array):
            sample = self.__pickle_file.iloc[j]

            # Pre process the image.
            img = load_img(io.BytesIO(sample["image"]), target_size=self.__image_size)
            x = img_to_array(img)
            x = self.__image_data_generator.random_transform(x)
            x = self.__image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.__labelled:
                batch_y[i, sample["label"]] = 1
                if self.__use_hierarchical_label:
                    batch_y_level1[i, sample["label_level1"]] = 1
                    batch_y_level2[i, sample["label_level2"]] = 1

        if self.__labelled:
            if not self.__use_hierarchical_label:
                return batch_x, batch_y
            else:
                return batch_x, [batch_y_level1, batch_y_level2, batch_y]
        else:
            return batch_x


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

num_gpus = 8
batch_size = 256*num_gpus


# Create a generator for training and a generator for validation.
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen = BSONIterator(train_bson_file,
                         train_image_table,
                         train_product_table,
                         None,
                         train_datagen,
                         batch_size=batch_size,
                         labelled=False,
                         )

val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = PickleGenerator(None,
                          pickle_file,
                          val_datagen,
                          batch_size=batch_size,
                          labelled=False,
                          )


# ---------------------------------------------------------------------------------
# Inference on multi-GPU
#

with tf.device("/cpu:0"):
    model = xception(None, trainable_layers=126, use_darc1=False, as_extractor=True)

# make the model parallel
parallel_model = multi_gpu_model(model, gpus=num_gpus)

#model.summary()

bottleneck_features_train = parallel_model.predict_generator(
    train_gen,
    steps=(num_train_images // batch_size)+1,
    workers=4,
    verbose=1
)
print(2)
np.save(utils_dir+'bottleneck_features_train.npy', bottleneck_features_train)
print("Successfully save bottleneck_features_train")
"""
bottleneck_features_val = parallel_model.predict_generator(
    val_gen,
    steps=(num_val_images // batch_size)+1,
    workers=4,
    verbose=1
)

np.save(utils_dir+'bottleneck_features_val.npy', bottleneck_features_val)
print("Successfully save bottleneck_features_val")
"""