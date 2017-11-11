"""
Define models
"""
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Dropout, Flatten, BatchNormalization, Dense, Activation
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD, Adamax
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
import utils.utils as utils


num_classes = utils.num_classes
num_classes1 = utils.num_class_level_one
num_classes2 = utils.num_class_level_two


# define loss function DARC1, employed by paper: "Generalization in Deep Learning"
# please set the model's output without "Softmax"
def DARC1(y_true, y_pred):
    y_pred_softmax = K.softmax(y_pred)
    xentropy = K.categorical_crossentropy(y_true, y_pred_softmax)
    reg = K.max(K.sum(K.abs(y_pred), axis=0))
    alpha = 0.001
    return xentropy+alpha*reg


def branch(input, num_filters, num_output, name, actvation='relu'):
    # input = MaxPooling2D((3, 3), strides=(2, 2), name=name+'_max_pool')(input)
    b = SeparableConv2D(num_filters, (3, 3), padding='same', use_bias=False, name=name+'_sepconv1')(input)
    b = BatchNormalization(name=name+'_sepconv1_bn')(b)
    b = Activation(actvation, name=name+'_sepconv1_act')(b)
    b = SeparableConv2D(num_filters, (3, 3), padding='same', use_bias=False, name=name+'_sepconv2')(b)
    b = BatchNormalization(name=name+'_sepconv2_bn')(b)
    b = Activation(actvation, name=name+'_sepconv2_act')(b)
    b = GlobalAveragePooling2D(name=name+'_avg_pool')(b)
    # output (no softmax for DARC1)
    output = Dense(num_output, name=name)(b)
    return output


# Xception model
def xception(num_dense, lr, use_pretrain=True, trainable_layers=126):
    if not use_pretrain:
        model = Xception(include_top=True, weights=None, input_shape=(180, 180, 3), classes=num_classes)
    else:
        base_model = Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # add a fully connected layer to reduce the parameters between last 2 layers
#        x = Dense(num_dense)(x)
#        x = BatchNormalization()(x)
#        x = Activation("relu")(x)
        # add a logistic layer
        output = Dense(num_classes)(x)
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=output)

        # optimizer
        adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        # record metadata during training
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        # we chose to train the top block and dense layers, i.e. we will freeze
        # the first 126 layers and unfreeze the rest:
        for layer in model.layers[:trainable_layers]:
            layer.trainable = False
        for layer in model.layers[trainable_layers:]:
            layer.trainable = True

        model.compile(optimizer=adam,
                      loss=DARC1,
                      metrics=["accuracy"],
                      #              options=run_options,
                      #              run_metadata=run_metadata,
                      )
    return model


def xception_branch(num_dense):
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))

    # add branches into the model

    # branch1
    x1 = base_model.layers[106].output
    output1 = branch(x1, 728, num_classes1, name='b1')

    # branch2
    x2 = base_model.layers[116].output
    output2 = branch(x2, 728, num_classes2, name='b2')

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add a fully connected layer to reduce the parameters between last 2 layers
    x = Dense(num_dense)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # output (no softmax for DARC1)
    output3 = Dense(num_classes, name="out")(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=[output1, output2, output3])

    # we chose to train the top block and dense layers, i.e. we will freeze
    # the first 126 layers and unfreeze the rest:
    for layer in model.layers[:86]:
        layer.trainable = False
    for layer in model.layers[86:]:
        layer.trainable = True

    return model


def resnet_50(lr):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(180, 180, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes)(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=output)

    # optimizer
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # record metadata during training
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    # we chose to train the top block and dense layers, i.e. we will freeze
    # the first 126 layers and unfreeze the rest:
    for layer in model.layers[:0]:
        layer.trainable = False
    for layer in model.layers[0:]:
        layer.trainable = True

    model.compile(optimizer=adam,
                  loss=DARC1,
                  metrics=["accuracy"],
                  #              options=run_options,
                  #              run_metadata=run_metadata,
                  )

    return model
