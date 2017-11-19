"""
Fine-tuning last several layers for the model.

Zhimeng Luo
"""
import os
import io
import math
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
from keras.optimizers import Adam, SGD, Adamax
from keras.layers import Lambda, BatchNormalization, Dense, Activation, Input
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, TensorBoard, ModelCheckpoint

# custom callbacks, not the original keras one
from utils.callbacks import SnapshotModelCheckpoint
import utils.utils as utils

from model import *
from keras.models import Model

data_dir = utils.data_dir
utils_dir = utils.utils_dir
log_dir = utils.log_dir
model_dir = utils.model_dir

train_image_table = pd.read_csv(utils.utils_dir + "train_images.csv", index_col=0)
val_image_table = pd.read_csv(utils.utils_dir + "val_images.csv", index_col=0)

num_train_images = len(train_image_table)
num_val_images = len(val_image_table)
num_classes = [utils.num_classes, utils.num_class_level_one, utils.num_class_level_two]

initial_learning_rate = 0.001
momentum = 0.9
decay_value = 0.1
decay_epoch = 2
batch_size = 1024
num_epoch = 50
num_final_dense_layer = 128
model_prefix = 'Xception-pretrained-%d' % num_final_dense_layer

# Load bottleneck features
train_data = np.load(utils_dir+'bottleneck_features_train.npy')
val_data = np.load(utils_dir+'bottleneck_features_val.npy')

# Load labels
train_label = np.array(train_image_table)[:, 1]
val_label = np.array(val_image_table)[:, 1]


# learning rate schedule (exponential decay)
def exp_decay(t):
    lrate = initial_learning_rate * math.pow(decay_value, math.floor((1+t)/decay_epoch))
    return lrate


def model_last_block(input_shape, num_dense, use_darc1=False):
    #base_model = Xception(include_top=False, weights=None, input_shape=(180,180,3),classes=num_classes[2])
    #output = []
    #input = model.input
    """
    for i in range(2):
    b = SeparableConv2D(num_filters, (3,3), padding='same', use_bias=False, name='b'+str(i+1)+'_sepconv1')(input)
    b = BatchNormalization(name='b'+str(i+1)+'sepconv1_bn')(b)
    b = Activation(actvation, name='b'+str(i+1)+'sepconv1_act')(b)
    b = SeparableConv2D(num_filters, (3,3), padding='same', use_bias=False, name='b'+str(i+1)+'_sepconv2')(b)
    b = BatchNormalization(name='b'+str(i+1)+'sepconv2_bn')(b)
    b = Activation(actvation, name='b'+str(i+1)+'sepconv2_act')(b)
    output.append(Dense(num_classes[i], name='b'+str(i+1))(b))
    """
    #x = Dense(128,input_shape=train_data.shape[1:])(input)
    #x = BatchNormalization()(x)
    #x = Activation("relu")(x)
    #output.append(Dense(num_classes[2], name="out")(x))
    #output = Dense(num_classes[2], name="out")(x)
    #model = Model(inputs=base_model.input, outputs=[output[0], output[1], output[2]])
    #model = Model(inputs=model.input, outputs=output)

    inputs = Input(shape=(input_shape,))
    x = Dense(num_dense)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # add a output layer
    if use_darc1:
        output = Dense(num_classes[0])(x)
    else:
        output = Dense(num_classes[0], activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=inputs, outputs=output)
    return model


model = model_last_block(2048, num_final_dense_layer, False)

adam = Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=initial_learning_rate, momentum=momentum)

model.compile(
    optimizer=sgd,
#    loss=DARC1,
    loss='sparse_categorical_crossentropy',
#    loss_weight=[lw1, lw2, lw3],
    metrics=["accuracy"],
)

# Learning rate schedual
lr_scheduler = LearningRateScheduler(exp_decay)

# Visualization when training
tensorboard = TensorBoard(
    log_dir=log_dir+"/add_fc/%d" % num_final_dense_layer,
    histogram_freq=0,
    batch_size=batch_size,
    write_graph=True,
    write_images=False)

ModelCheckpoint(
    model_dir + "%s-{epoch:02d}-{val_acc:.3f}.h5" % model_prefix,
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=True,
)

model.fit(
    train_data,
    train_label,
    epochs=num_epoch,
    batch_size=batch_size,
    validation_data=(val_data, val_label),
    shuffle=True,
    callbacks=[
        tensorboard,
        lr_scheduler,
    ]
)
