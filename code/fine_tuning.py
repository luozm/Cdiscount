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
from model import *


# Load bottleneck features
train_data = np.load(open('bottleneck_features_train.npy'))
val_data = np.load(open('bottleneck_features_val.npy'))

# Load labels
train_label = np.zeros((len(batch_x), self.__num_label), dtype=K.floatx())
train_label_level1 = np.zeros((len(batch_x), self.__num_label_level1), dtype=K.floatx())
train_label_level2 = np.zeros((len(batch_x), self.__num_label_level2), dtype=K.floatx())

val_label = np.zeros((len(batch_x), self.__num_label), dtype=K.floatx())
val_label_level1 = np.zeros((len(batch_x), self.__num_label_level1), dtype=K.floatx())
val_label_level2 = np.zeros((len(batch_x), self.__num_label_level2), dtype=K.floatx())




# Last several layers to be fune-tuned
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

