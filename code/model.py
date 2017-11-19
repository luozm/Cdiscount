"""
Define models
"""
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dropout, Flatten, BatchNormalization, Dense, Activation
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD, Adamax
from keras.applications.xception import Xception
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
