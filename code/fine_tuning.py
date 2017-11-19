"""
Fine-tuning last several layers for the model.

Zhimeng Luo
"""
import os
import io
import math
import random
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
from keras.optimizers import Adam, SGD
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
result_dir = utils.result_dir

train_image_table = pd.read_csv(utils.utils_dir + "train_images.csv", index_col=0)
val_image_table = pd.read_csv(utils.utils_dir + "val_images.csv", index_col=0)

num_classes = [utils.num_classes, utils.num_class_level_one, utils.num_class_level_two]

decay_value = 0.1
decay_epoch = 2
num_epoch = 5
#num_final_dense_layer = 128
#model_prefix = 'Xception-pretrained-%d' % num_final_dense_layer

# Load bottleneck features
train_data = np.load(utils_dir+'bottleneck_features_train.npy')
val_data = np.load(utils_dir+'bottleneck_features_val.npy')

# Load labels
train_label = np.array(train_image_table)[:, 1]
val_label = np.array(val_image_table)[:, 1]

"""
# learning rate schedule (exponential decay)
def exp_decay(t):
    lrate = initial_learning_rate * math.pow(decay_value, math.floor((1+t)/decay_epoch))
    return lrate
"""


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


"""
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
"""


def train_random_search(max_val=10):
    result_list = []

    for i in range(max_val):
        # Params to be validated
        initial_learning_rate = 10**random.uniform(-2, -5)
#        initial_learning_rate = 0.003
        momentum = 1-10**random.uniform(0, -4)
#        momentum = 0.99
#        batch_size = random.choice([64, 256, 1024])
        batch_size = 512
        num_final_dense_layer = random.choice([32, 128, 512, 2048])

        model = model_last_block(2048, num_final_dense_layer, False)
        adam = Adam(lr=initial_learning_rate)
#        sgd = SGD(lr=initial_learning_rate, momentum=momentum)

        model.compile(
            optimizer=adam,
            #    loss=DARC1,
            loss='sparse_categorical_crossentropy',
            #    loss_weight=[lw1, lw2, lw3],
            metrics=["accuracy"],
        )

        # Learning rate scheduler
    #    lr_scheduler = LearningRateScheduler(exp_decay)

        model.fit(
            train_data[:50000,:],
            train_label[:50000],
#            train_data,
#            train_label,
            epochs=num_epoch,
            batch_size=batch_size,
            shuffle=True,
#            verbose=0,
    #        callbacks=[lr_scheduler]
        )

        _, acc = model.evaluate(
            train_data[:50000,:],
            train_label[:50000],
#            val_data,
#            val_label,
            batch_size=4096,
            verbose=0)

        result_row = {
            "idx": i+1,
            "train_acc": acc,
            "batch": batch_size,
            "num_dense": num_final_dense_layer,
            "lr": initial_learning_rate,
            "mom": momentum}
        result_list.append(result_row)
        print(result_row)

    result_df = pd.DataFrame(result_list)
    result_df.to_csv(utils.result_dir + "fine_tuning_result_adam.csv")


train_random_search(100)
