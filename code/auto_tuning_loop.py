import os
import io
import numpy as np
import pandas as pd

from keras.optimizers import Adam, SGD
from keras.layers import BatchNormalization, Dense, Activation, Input
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.utils.training_utils import multi_gpu_model

# custom callbacks, not the original keras one
from utils.callbacks import SnapshotModelCheckpoint
import utils.utils as utils

from model import *
from keras.models import Model, Sequential


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

hyper_param = utils.get_hyper_parameter('xception')


def model_last_block(final_dense_layer, num_classes):
    model = Sequential()
    model.add(Dense(final_dense_layer, input_shape=(2048,)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(num_classes[0], activation="softmax"))
    #inputs = Input(shape=(2048,))  # arguments['input_shape'],))
    #x = Dense(final_dense_layer)(input)
    #x = BatchNormalization()(x)
    #x = Activation("relu")(x)
    # add a output layer
    #    if conditional({{choice([True, False])}}):
    #        output = Dense(arguments['num_classes'][0])(x)
    #    else:
    #        output = Dense(arguments['num_classes'][0], activation='softmax')(x)
    #output = Dense(num_classes[0], activation='softmax')(x)
    # this is the model we will train
    #model = Model(inputs=inputs, outputs=output)
    return model

best_acc = 0

for final_dense_layer in hyper_param['model']['final_dense_layer']:
    for lr in hyper_param['optimizer']['sgd']['initial_learning_rate']:
        for momentum in hyper_param['optimizer']['sgd']['momentum']:
            for decay in hyper_param['optimizer']['sgd']['decay_value']:
                for batch_size in hyper_param['batch_size']:
                    #with tf.device("/cpu:0"):
                    #    model = model_last_block(final_dense_layer, num_classes)
                    model = model_last_block(final_dense_layer, num_classes)
                    #parallel_model = multi_gpu_model(model, gpus=8)
                    #    adam = Adam(lr={{choice(hyper_param['optimizer']['adam']['initial_learning_rate'])}},
                    #                beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                    #                decay={{choice(hyper_param['optimizer']['adam']['decay_value'])}})

                    sgd = SGD(lr=lr,
                              momentum=momentum,
                              decay=decay)

                    model.compile(
                        #        loss={{choice(['sparse_categorical_crossentropy', DARC1])}},
                        loss='sparse_categorical_crossentropy',
                        #        optimizer={{choice([sgd, adam])}},
                        optimizer=sgd,
                        metrics=["accuracy"],
                    )

                    model.fit(train_data, train_label,
                              epochs=10,  # arguments['num_epoch'],
                              batch_size=batch_size,
                              validation_data=(val_data, val_label),
                              shuffle=True,
                              callbacks=[reduce_lr]
                              )
                    score, acc = model.evaluate(val_data, val_label, verbose=0)
                    if acc > best_acc:
                        best_hp = {
                            'final_dense_layer': final_dense_layer,
                            'lr': lr,
                            'momentum': momentum,
                            'decay': decay,
                            'batch_size': batch_size,
                        }
                    print('Validation accuracy:', acc)

print(best_hp)
