"""
Fine-tuning last several layers for the model.

Zhimeng Luo
"""
import os
import io
import numpy as np
import pandas as pd
import bson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam, SGD, Adamax
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

# custom callbacks, not the original keras one
from utils.callbacks import TensorBoard, SnapshotModelCheckpoint, ModelCheckpoint
import utils.utils as utils
from keras.layers import Lambda, BatchNormalization, Dense, Activation
from model import *
from keras.models import Model, Sequential


# Load bottleneck features
#train_data = np.load(open('bottleneck_features_train.npy'))
train_data = np.zeros((11139926,1))
#val_data = np.load(open('bottleneck_features_val.npy'))
val_data = np.zeros((1231367,1))

# Load labels
train_image_table = pd.read_csv(utils.utils_dir + "train_images.csv", index_col=0)
train_label = np.array(train_image_table)[:,1]
#train_label_level1 = np.zeros((len(batch_x), self.__num_label_level1), dtype=K.floatx())
#train_label_level2 = np.zeros((len(batch_x), self.__num_label_level2), dtype=K.floatx())

val_image_table = pd.read_csv(utils.utils_dir + "val_images.csv", index_col=0)
val_label = np.array(val_image_table)[:,1]
#val_label_level1 = np.zeros((len(batch_x), self.__num_label_level1), dtype=K.floatx())
#val_label_level2 = np.zeros((len(batch_x), self.__num_label_level2), dtype=K.floatx())


def model(shape,num_classes):#, num_filters):
    #base_model = Xception(include_top=False, weights=None, input_shape=(180,180,3),classes=num_classes[2])
    model = Sequential()
    #output = []
    #input = model.input
    '''for i in range(2):
        b = SeparableConv2D(num_filters, (3,3), padding='same', use_bias=False, name='b'+str(i+1)+'_sepconv1')(input)
        b = BatchNormalization(name='b'+str(i+1)+'sepconv1_bn')(b)
        b = Activation(actvation, name='b'+str(i+1)+'sepconv1_act')(b)
        b = SeparableConv2D(num_filters, (3,3), padding='same', use_bias=False, name='b'+str(i+1)+'_sepconv2')(b)
        b = BatchNormalization(name='b'+str(i+1)+'sepconv2_bn')(b)
        b = Activation(actvation, name='b'+str(i+1)+'sepconv2_act')(b)
        output.append(Dense(num_classes[i], name='b'+str(i+1))(b))'''
    #x = Dense(128,input_shape=train_data.shape[1:])(input)
    #x = BatchNormalization()(x)
    #x = Activation("relu")(x)
    #output.append(Dense(num_classes[2], name="out")(x))
    #output = Dense(num_classes[2], name="out")(x)
    #model = Model(inputs=base_model.input, outputs=[output[0], output[1], output[2]])
    #model = Model(inputs=model.input, outputs=output)
    model.add(Dense(128,input_shape=shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(num_classes[2]))
    return model

num_classes = [utils.num_classes, utils.num_class_level_one, utils.num_class_level_two]

model = model(train_data.shape[1:],num_classes)#, 728)

initial_learning_rate = 0.001
adam = Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam,
              loss=DARC1,
              #loss_weight=[lw1, lw2, lw3],
              metrics=["accuracy"],
              )
model.fit(train_data, train_label,
          epochs=50,
          batch_size=128,
          validation_data=(val_data, val_label))