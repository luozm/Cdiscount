import os
import io
import numpy as np
import pandas as pd

from keras.optimizers import Adam, SGD
from keras.layers import BatchNormalization, Dense, Activation, Input
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, TensorBoard, ModelCheckpoint

# custom callbacks, not the original keras one
from utils.callbacks import SnapshotModelCheckpoint
import utils.utils as utils

from model import *
from keras.models import Model

import json

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional


def data():
    utils_dir = utils.utils_dir
    log_dir = utils.log_dir
    model_dir = utils.model_dir

    train_image_table = pd.read_csv(utils.utils_dir + "train_images.csv", index_col=0)
    val_image_table = pd.read_csv(utils.utils_dir + "val_images.csv", index_col=0)

    num_classes = [utils.num_classes, utils.num_class_level_one, utils.num_class_level_two]
    num_epoch = 100
    num_final_dense_layer = 128
    model_prefix = 'Xception-pretrained-%d' % num_final_dense_layer

    train_data = np.load(utils_dir + 'bottleneck_features_train.npy')
    val_data = np.load(utils_dir + 'bottleneck_features_val.npy')

    train_label = np.array(train_image_table)[:, 1]
    val_label = np.array(val_image_table)[:, 1]

    modelcheckpoint = ModelCheckpoint(
        model_dir + "%s-{epoch:02d}-{val_acc:.3f}.h5" % model_prefix,
        monitor="val_acc",
        save_best_only=True,
        save_weights_only=True,
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

    arguments = {
        'input_shape': 2048,
        'num_epoch': num_epoch,
        'modelcheckpoint': modelcheckpoint,
        'reduce_lr': reduce_lr,
        'num_classes': num_classes,
        'use_darc1': False,
        'log_dir': log_dir,

    }

    hyperpara = utils.get_hyper_parameter('xception')

    return train_data, train_label, val_data, val_label, arguments, hyperpara


def create_model(train_data, train_label, val_data, val_label, arguments, hyperpara):
    inputs = Input(shape=(arguments['input_shape'],))
    x = Dense({{choice(hyperpara['model']['final_dense_layer'])}})(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # add a output layer
    if conditional({{choice([True, False])}}):
        output = Dense(arguments['num_classes'][0])(x)
    else:
        output = Dense(arguments['num_classes'][0], activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=inputs, outputs=output)

    adam = Adam(lr={{choice(hyperpara['optimizer']['adam']['initial_learning_rate'])}},
                beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                decay={{choice(hyperpara['optimizer']['adam']['decay_value'])}})

    sgd = SGD(lr={{choice(hyperpara['optimizer']['sgd']['initial_learning_rate'])}},
              momentum={{choice(hyperpara['optimizer']['sgd']['momentum'])}},
              decay={{choice(hyperpara['optimizer']['sgd']['decay_value'])}})

    model.compile(
        loss={{choice(['sparse_categorical_crossentropy', DARC1])}},
        optimizer={{choice([sgd, adam])}},
        metrics=["accuracy"],
    )

    tensorboard = TensorBoard(
        log_dir=arguments['log_dir'] + "/add_fc/%d" % conditional({{choice(hyperpara['model']['final_dense_layer'])}}),
        histogram_freq=0,
        batch_size={{choice(hyperpara['batch_size'])}},
        write_graph=True,
        write_images=False
    )

    model.fit(train_data, train_label,
              epochs=arguments['num_epoch'],
              batch_size={{choice(hyperpara['batch_size'])}},
              validation_data=(val_data, val_label),
              shuffle=True,
              callbacks=[tensorboard, arguments['modelcheckpoint'], arguments['reduce_lr']]
              )
    score, acc = model.evaluate(val_data, val_label, verbose=0)
    print('Validation accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


best_run, best_model = optim.minimize(
    model=create_model,
    data=data,
    algo=tpe.suggest,
    max_evals=5,
    trials=Trials()
)

X_train, Y_train, X_test, Y_test, _, _ = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)
