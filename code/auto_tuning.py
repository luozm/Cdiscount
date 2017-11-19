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
    num_epoch = 10

    train_data = np.load(utils_dir + 'bottleneck_features_train.npy')
    val_data = np.load(utils_dir + 'bottleneck_features_val.npy')

    train_label = np.array(train_image_table)[:, 1]
    val_label = np.array(val_image_table)[:, 1]

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1)

    arguments = {
        'input_shape': 2048,
        'num_epoch': num_epoch,
        'reduce_lr': reduce_lr,
        'num_classes': num_classes,
        'use_darc1': False,
        'log_dir': log_dir,
        'model_dir': model_dir,
    }

    hyper_param = utils.get_hyper_parameter('xception')

    return train_data, train_label, val_data, val_label, reduce_lr, num_classes#arguments, hyper_param


def create_model(train_data, train_label, val_data, val_label, reduce_lr, num_classes):#arguments, hyper_param):
    inputs = Input(shape=(2048,))#arguments['input_shape'],))
    x = Dense({{choice([128, 1024])}})(input)#hyper_param['model']['final_dense_layer'])}})(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # add a output layer
#    if conditional({{choice([True, False])}}):
#        output = Dense(arguments['num_classes'][0])(x)
#    else:
#        output = Dense(arguments['num_classes'][0], activation='softmax')(x)
    output = Dense(num_classes[0], activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=inputs, outputs=output)

#    adam = Adam(lr={{choice(hyper_param['optimizer']['adam']['initial_learning_rate'])}},
#                beta_1=0.9, beta_2=0.999, epsilon=1e-08,
#                decay={{choice(hyper_param['optimizer']['adam']['decay_value'])}})

    sgd = SGD(lr={{choice([0.01, 0.001, 0.0001])}},#hyper_param['optimizer']['sgd']['initial_learning_rate'])}},
              momentum={{choice([0.95, 0.9, 0.85, 0.8])}},#hyper_param['optimizer']['sgd']['momentum'])}},
              decay={{choice([0.1, 0.3, 0.5])}})#hyper_param['optimizer']['sgd']['decay_value'])}})

    model.compile(
#        loss={{choice(['sparse_categorical_crossentropy', DARC1])}},
        loss='sparse_categorical_crossentropy',
#        optimizer={{choice([sgd, adam])}},
        optimizer=sgd,
        metrics=["accuracy"],
    )

    model.fit(train_data, train_label,
              epochs=10,#arguments['num_epoch'],
              batch_size={{choice([256, 2048])}},#hyper_param['batch_size'])}},
              validation_data=(val_data, val_label),
              shuffle=True,
              callbacks=[reduce_lr]
              )
    score, acc = model.evaluate(val_data, val_label, verbose=0)
    print('Validation accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


X_train, Y_train, X_test, Y_test, _, _ = data()

best_run, best_model = optim.minimize(
    model=create_model,
    data=data,
    algo=tpe.suggest,
    max_evals=10,
    trials=Trials()
)

print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)
