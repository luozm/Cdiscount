# This notebook contains a generator class for
# Keras called `BSONIterator` that can read directly from the BSON data.
# You can use it in combination with `ImageDataGenerator` for doing data augmentation.

from importlib import reload
import utils.sysmonitor as SM
reload(SM)
import os
import io
import numpy as np
import pandas as pd
import bson
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline

import tensorflow as tf
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Dropout, Flatten, BatchNormalization, Dense, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD, Adamax
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

# custom callbacks, not the original keras one
from utils.callbacks import TensorBoard, SnapshotModelCheckpoint, LossWeightsModifier
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
        print("Found %d images belonging to %d classes." % (self.__num_images, self.__num_label))

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

##
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
        print("Found %d images belonging to %d classes." % (self.__num_images, self.__num_label))

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


# visualizing losses and accuracy, and real learning rate
def visual_result(hist, lr):
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    xc = range(num_epoch)

    # Losses
    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])

    # use bmh, classic,ggplot for big pictures
    plt.style.available
    plt.style.use(['classic'])

    # Accuracy
    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    # use bmh, classic,ggplot for big pictures
    plt.style.available
    plt.style.use(['classic'])

    # LearningRate
    plt.figure(3, figsize=(7, 5))
    plt.plot(xc, lr)
    plt.xlabel('num of Epochs')
    plt.ylabel('learning rate')
    plt.title('Real Learning Rate')
    plt.grid(True)
    plt.legend(['lr'])

    plt.show()


# LearningRate Schedule
def _cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (num_epoch // num_snapshots))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= num_epoch // num_snapshots
    cos_out = np.cos(cos_inner) + 1
    return float(initial_learning_rate / 2 * cos_out)


# Part 2: The generator
# Input data files are available in the "../data/input/" directory.
data_dir = utils.data_dir
utils_dir = utils.utils_dir
log_dir = utils.log_dir
model_dir = utils.model_dir
train_bson_path = os.path.join(data_dir, "train.bson")

# First load the lookup tables from the CSV files.
train_product_table = pd.read_csv(utils_dir + "train_offsets.csv", index_col=0)
train_image_table = pd.read_csv(utils_dir + "train_images.csv", index_col=0)
pickle_file = pd.read_pickle(utils_dir + "val_dataset.pkl")

num_classes = utils.num_classes
num_classes_level1 = utils.num_class_level_one
num_classes_level2 = utils.num_class_level_two
num_train_images = len(train_image_table)
num_val_images = len(pickle_file)
train_bson_file = open(train_bson_path, "rb")

batch_size = 256
num_epoch = 6
initial_learning_rate = 0.001
num_final_dense_layer = 128
num_snapshots = 1
model_prefix = 'Xception-nofc-pretrained-%d' % num_final_dense_layer


# Create a generator for training and a generator for validation.

# Tip: use ImageDataGenerator for data augmentation and preprocessing.
train_datagen = ImageDataGenerator(
     rescale=1./255,
     horizontal_flip=True,
     zoom_range=0.2,
     rotation_range=20,
     width_shift_range=0.2,
     height_shift_range=0.2,
      )
train_gen = BSONIterator(train_bson_file,
                         train_image_table,
                         train_product_table,
                         num_classes,
                         train_datagen,
                         batch_size=batch_size,
                         shuffle=True,
                         )

val_datagen = ImageDataGenerator(
    rescale=1./255
)
val_gen = PickleGenerator(num_classes,
                          pickle_file,
                          val_datagen,
                          batch_size=batch_size,
                          )

"""
# show images after augmentation
count = 0
n = 4
while count <= 15:
    bx, by = next(train_gen)
    if count % n == 0:
        plt.figure(figsize=(14, 6))
    plt.subplot(1, n, count % n + 1)
    plt.imshow(bx[-1].astype(np.uint8))
    plt.axis('off')
    count += 1
plt.show()
"""

# Part 3: Training
model = xception(1, initial_learning_rate)


# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(model.layers):
#    print(i, layer.name)


# Train a Xception without pre-train parameters
# model = Xception(include_top=True, weights=None, input_shape=(180, 180, 3), classes=num_classes)

# model = load_model('./weights/Xception-pretrained-128-Best.h5')

model.summary()

# Callbacks
# Visualization when training
tensorboard = TensorBoard(
    log_dir=log_dir+"/nofc",
    histogram_freq=0,
    batch_size=batch_size,
    write_graph=True,
    write_images=False,
    write_batch_performance=True)


# Reduce learning rate when loss has stopped improving
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=2)

# Build snapshot callback
snapshot = SnapshotModelCheckpoint(num_epoch, num_snapshots, fn_prefix=model_dir + "%s" % model_prefix)

callback_list = [
    ModelCheckpoint(
        model_dir+"%s-Best.h5" % model_prefix,
        monitor="val_acc",
        save_best_only=True,
        save_weights_only=False),
    LearningRateScheduler(schedule=_cosine_anneal_schedule),
    snapshot,
#    LossWeightsModifier(lw1, lw2, lw3),
    tensorboard,
]

# Monitoring the status of GPU & CPU
sys_mon = SM.SysMonitor()
sys_mon.start()


# To train the model:
history = model.fit_generator(
    train_gen,
    steps_per_epoch=(num_train_images // batch_size)+1,
    epochs=num_epoch,
    validation_data=val_gen,
    validation_steps=(num_val_images // batch_size)+1,
    max_queue_size=100,
    workers=8,
    callbacks=callback_list,
    # initial_epoch=2
    )

sys_mon.stop()
title = '{0:.2f} seconds of computation, no multiprocessing, batch size = {1}'.format(sys_mon.duration, batch_size)
sys_mon.plot(title, True)
plt.show()

print(snapshot.lr)
visual_result(history, snapshot.lr)

"""
# Profling using timeline (go to the page chrome://tracing and load the timeline.json file)
trace = timeline.Timeline(step_stats=run_metadata.step_stats)
with open('timeline.ctf.json', 'w') as f:
    f.write(trace.generate_chrome_trace_format())


# Save model
savepath='./models/Xception_model.h5'
model.save(savepath)
print("Successfully save model in Xception_model.h5")
"""
