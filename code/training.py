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
import threading
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
from utils.callbacks import TensorBoard, SnapshotModelCheckpoint
import utils.utils as utils


# The Keras generator is implemented by the `BSONIterator` class.
# It creates batches of images (and their one-hot labels) directly from the BSON file.
# It can be used with multiple workers.
#
# See also the code in: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
class BSONIterator(Iterator):
    def __init__(self, bson_file, image_table, product_table, num_label,
                 image_data_generator, lock, image_size=(180, 180), labelled=True,
                 batch_size=32, shuffle=False, seed=None):
        """
        :param bson_file:       The original bson file instance
        :param image_table:     The table that stores information of each image
        :param product_table:   The table that stores position of each product
        :param num_label:       The amount of categories
        :param image_data_generator: Data augmentation generator
        :param lock:            File locker
        :param image_size:      The size of image
        :param labelled:        Check if the image is of validation or training set
        :param batch_size:      Size of the batch
        :param shuffle:         Whether use the shuffle strategy
        :param seed:            Random seed
        """
        # Parameter initialization
        self.__file = bson_file
        self.__image_table = image_table
        self.__product_table = product_table
        self.__labelled = labelled
        self.__num_images = len(image_table)
        self.__num_label = num_label
        self.__image_data_generator = image_data_generator
        self.__image_size = tuple(image_size)
        self.__image_shape = self.__image_size + (3,)
        self.__lock = lock
        # Pass parameter back to super class
        super(BSONIterator, self).__init__(self.__num_images, batch_size, shuffle, seed)

        # Print out information
        print("Found %d images belonging to %d classes." % (self.__num_images, self.__num_label))

    def _get_next_batch(self, index_array):
        """
        :param index_array:
        :return: The batch of samples(pair of image and label)
        """
        batch_x = np.zeros((len(index_array),) + self.__image_shape, dtype=K.floatx())
        if self.__labelled:
            batch_y = np.zeros((len(batch_x), self.__num_label), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and data frame access with a lock.
            with self.__lock:
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

        if self.__labelled:
            return batch_x, batch_y
        else:
            return batch_x

    def next(self):
        with self.__lock:
            index_array = next(self.index_generator)
        return self._get_next_batch(index_array)


class PickleGenerator(Iterator):
    def __init__(self, num_label, pickle_file, image_data_generator, batch_size=32,
                 image_size=(180, 180), labelled=True, shuffle=False, seed=None,):
        """
        :param pickle_file:       The original bson file instance
        :param num_label:       The amount of categories
        :param image_data_generator: Data augmentation generator
        :param image_size:      The size of image
        :param labelled:        Check if the image is of validation or training set
        :param batch_size:      Size of the batch
        :param shuffle:         Whether use the shuffle strategy
        :param seed:            Random seed
        """
        self.__pickle_file = pickle_file
        self.__batch_size = batch_size
        self.__labelled = labelled
        self.__num_label = num_label
        self.__image_data_generator = image_data_generator
        self.__image_size = tuple(image_size)
        self.__image_shape = self.__image_size + (3,)
        self.__num_images = len(pickle_file)

        # pass arguments to super class
        super(PickleGenerator, self).__init__(self.__num_images, batch_size, shuffle, seed)
        # Print out information
        print("Found %d images belonging to %d classes." % (self.__num_images, self.__num_label))

    def next(self):
        index_array = next(self.index_generator)
        return self._get_next_batch(index_array)

    def _get_next_batch(self, index_array):
        """
        :param index_array:
        :return: Batch of samples(pair of image and label)
        """
        batch_x = np.zeros((len(index_array),) + self.__image_shape, dtype=K.floatx())
        if self.__labelled:
            batch_y = np.zeros((len(batch_x), self.__num_label), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and data frame access with a lock.
            with self.__lock:
                sample = self.__pickle_file.iloc[j]

            # Pre process the image.
            img = load_img(io.BytesIO(sample["x"]), target_size=self.__image_size)
            x = img_to_array(img)
            x = self.__image_data_generator.random_transform(x)
            x = self.__image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.__labelled:
                batch_y[i, sample["y"]] = 1

        if self.__labelled:
            return batch_x, batch_y
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
num_train_images = len(train_image_table)
num_val_images = len(pickle_file)
lock = threading.Lock()
train_bson_file = open(train_bson_path, "rb")

batch_size = 256
num_epoch = 6
initial_learning_rate = 0.001
num_final_dense_layer = 128
num_snapshots = 3
model_prefix = 'Xception-pretrained-%d' % num_final_dense_layer


# Create a generator for training and a generator for validation.

# Tip: use ImageDataGenerator for data augmentation and preprocessing.
train_datagen = ImageDataGenerator(
    # rescale=1./255,
    # horizontal_flip=True,
    # zoom_range=0.2,
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
      )
train_gen = BSONIterator(train_bson_file,
                         train_image_table,
                         train_product_table,
                         num_classes,
                         train_datagen,
                         lock,
                         batch_size=batch_size,
                         shuffle=True
                         )

val_datagen = ImageDataGenerator()
val_gen = PickleGenerator(pickle_file, val_datagen, batch_size=batch_size)

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

base_model = Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# add a fully connected layer to reduce the parameters between last 2 layers
x = Dense(num_final_dense_layer)(x)
x = BatchNormalization()(x)
x = Activation("selu")(x)

# add a logistic layer
output = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=output)


# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(model.layers):
#    print(i, layer.name)

# we chose to train the top block and dense layers, i.e. we will freeze
# the first 126 layers and unfreeze the rest:
for layer in model.layers[:126]:
    layer.trainable = False
for layer in model.layers[126:]:
    layer.trainable = True


# Train a Xception without pre-train parameters
# model = Xception(include_top=True, weights=None, input_shape=(180, 180, 3), classes=num_classes)

# model = load_model('./weights/Xception-pretrained-128-Best.h5')

# optimizer
adam = Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# record metadata during training
# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()

model.compile(optimizer=adam,
              loss="categorical_crossentropy",
              metrics=["accuracy"],
              # options=run_options,
              # run_metadata=run_metadata,
              )

model.summary()

# Callbacks
# Visualization when training
tensorboard = TensorBoard(log_dir=log_dir+"/add_fc/%d" % num_final_dense_layer,
                          histogram_freq=0,
                          batch_size=batch_size,
                          write_graph=True,
                          write_images=False,
                          write_batch_performance=True)

# Save model for every epoch
# checkpoint = ModelCheckpoint("./models/Xception_{epoch:02d}_{val_loss:.2f}.h5")

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
    snapshot, tensorboard]

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