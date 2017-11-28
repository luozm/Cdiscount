# This notebook contains a generator class for
# Keras called `BSONIterator` that can read directly from the BSON data.
# You can use it in combination with `ImageDataGenerator` for doing data augmentation.

import os
import pandas as pd

import tensorflow as tf
import keras
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

# custom callbacks, not the original keras one
from utils.callbacks import TensorBoard, SnapshotModelCheckpoint, ModelCheckpoint, LossWeightsModifier
from utils import utils
from model.xception import xception_3branch
from utils.iterator import PickleIterator
from utils.iterator import BSONIterator
from model.loss import darc1


# ---------------------------------------------------------------------------------
# Initialization
#

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

use_crop = utils.use_crop
num_gpus = 8
batch_size = 32*num_gpus
num_epoch = 10
initial_learning_rate = 0.0001
momentum = 0.65
decay_value = 0.1
alpha = 5e-5
num_snapshots = 1
model_prefix = 'Xception-combine-3branch'


# Create a generator for training and a generator for validation.

# Tip: use ImageDataGenerator for data augmentation and preprocessing.
train_datagen = ImageDataGenerator(
    samplewise_center=True,
    rescale=1./255,
    horizontal_flip=True,
#    zoom_range=0.2,
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
)
train_gen = BSONIterator(
    train_bson_file,
    train_image_table,
    train_product_table,
    num_classes,
    train_datagen,
    num_label_level1=num_classes_level1,
    num_label_level2=num_classes_level2,
    batch_size=batch_size,
    shuffle=True,
    use_hierarchical_label=True,
    use_crop=use_crop
)

val_datagen = ImageDataGenerator(
    samplewise_center=True,
    rescale=1./255
)
val_gen = PickleIterator(
    num_classes,
    pickle_file,
    val_datagen,
    batch_size=batch_size,
    num_label_level1=num_classes_level1,
    num_label_level2=num_classes_level2,
    use_hierarchical_label=True,
    use_crop=use_crop
)

# ---------------------------------------------------------------------------------
# Training on multi-GPU
#

with tf.device("/cpu:0"):
    model = xception_3branch(512, 1024, 2048, False)

    # Load weights
#    model.load_weights(model_dir+'%s-0.600.h5' % model_prefix)
    model.load_weights(model_dir+'%s-01-0.566.h5' % model_prefix)

# make the model parallel
parallel_model = multi_gpu_model(model, gpus=num_gpus)

# optimizer
sgd = SGD(lr=initial_learning_rate, momentum=momentum)

# loss weights of 3 branches
lw1 = K.variable(value=0.1, dtype="float32", name="loss_weight1")  # A1 in paper
lw2 = K.variable(value=0.2, dtype="float32", name="loss_weight2")  # A2 in paper
lw3 = K.variable(value=0.7, dtype="float32", name="loss_weight3")  # A3 in paper

parallel_model.compile(
    optimizer=sgd,
    loss=darc1(alpha),
    loss_weights=[lw1, lw2, lw3],
    metrics=["accuracy"],
)

model.summary()

# Callbacks

# Visualization when training
tensorboard = TensorBoard(
    log_dir=log_dir+"/combine/%f-%f-%f" % (initial_learning_rate, momentum, alpha),
    batch_size=batch_size,
    write_graph=True,
    write_batch_performance=True)

# Model checkpoints
checkpoint = ModelCheckpoint(
    model_dir + "%s-{epoch:02d}-{val_concatenate_3_acc:.3f}.h5" % model_prefix,
    save_best_only=False,
    save_weights_only=True,
    base_model=model
)
# Reduce learning rate when loss has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_concatenate_3_loss', factor=decay_value, patience=1, verbose=1)

# Build snapshot callback
snapshot = SnapshotModelCheckpoint(num_epoch, num_snapshots, model, fn_prefix=model_dir + "%s" % model_prefix)

callback_list = [
    checkpoint,
#    snapshot,
#    LossWeightsModifier(lw1, lw2, lw3),
    reduce_lr,
    tensorboard,
]

score = parallel_model.evaluate_generator(
    val_gen,
    steps=(10000 // batch_size)+1,
    workers=8)
print(score)

# To train the model:
history = parallel_model.fit_generator(
    train_gen,
    steps_per_epoch=(num_train_images // batch_size)+1,
    epochs=num_epoch,
    validation_data=val_gen,
    validation_steps=(num_val_images // batch_size)+1,
    workers=8,
    callbacks=callback_list,
    # initial_epoch=2
)

#print(snapshot.lr)
#visual_result(history, snapshot.lr)
