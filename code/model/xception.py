import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dropout, Flatten, BatchNormalization, Dense, Activation
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD, Adamax
from keras.applications.xception import Xception
from utils.utils import num_classes


# Xception model
def xception(num_dense, use_pretrain=True, trainable_layers=126, use_darc1=False ):
    if not use_pretrain:
        model = Xception(include_top=True, weights=None, input_shape=(180, 180, 3), classes=num_classes)
    else:
        base_model = Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # add a fully connected layer to reduce the parameters between last 2 layers
        x = Dense(num_dense)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # add a output layer
        if use_darc1:
            output = Dense(num_classes)(x)
        else:
            output = Dense(num_classes, activation='softmax')(x)
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=output)

        # we chose to train the top block and dense layers, i.e. we will freeze
        # the first 126 layers and unfreeze the rest:
        for layer in model.layers[:trainable_layers]:
            layer.trainable = False
        for layer in model.layers[trainable_layers:]:
            layer.trainable = True
    return model


def extractor_by_layer(layer):
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))
    # add a global spatial average pooling layer
    x = base_model.layers[layer].output
    x = GlobalAveragePooling2D()(x)

    # return extractor
    extractor_model = Model(inputs=base_model.input, outputs=x)
    return extractor_model

def extractor_by_name (name):
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))
    # add a global spatial average pooling layer
    x = base_model.get_layer(name).output
    x = GlobalAveragePooling2D()(x)

    # return extractor
    extractor_model = Model(inputs=base_model.input, outputs=x)
    return extractor_model