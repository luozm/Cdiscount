"""
Xception models

"""

from keras.models import Model
from keras.layers import Input, Dropout, Flatten, BatchNormalization, Dense, Activation
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.applications.xception import Xception

from utils import utils

num_classes = utils.num_classes
num_classes1 = utils.num_class_level_one
num_classes2 = utils.num_class_level_two


def xception(add_fc, num_units=None, use_pretrained_weights=True, trainable_layers=126, use_softmax=True):
    """Xception model

    Load Xception model from keras.applications.

    :param add_fc: whether or not add another fully connected layer before logits
    :param num_units: number of units used in additional fc layer
    :param use_pretrained_weights: whether or not use pre-trained weights on ImageNet
    :param trainable_layers: frozen first 'trainable_layers' layers, train other layers after it
    :param use_softmax: whether or not apply softmax to logits

    :return: loaded Xception model
    """
    if not use_pretrained_weights:
        model = Xception(include_top=True, weights=None, input_shape=(180, 180, 3), classes=num_classes)
    else:
        base_model = Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        if add_fc:
            assert num_units is not None

            # add a fully connected layer to reduce the parameters between last 2 layers
            x = Dense(num_units)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        # add a output layer
        if not use_softmax:
            output = Dense(num_classes)(x)
        else:
            output = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output)

        # we chose to train the top block and dense layers, i.e. we will freeze
        # the first 'trainable_layers' layers and unfreeze the rest:
        for layer in model.layers[:trainable_layers]:
            layer.trainable = False
        for layer in model.layers[trainable_layers:]:
            layer.trainable = True

    return model


def extractor_3outputs(layer1, layer2, layer3):
    """Bottleneck feature extractor (both 3 outputs)

    Extract bottleneck feature used for fine-tuning last several layers.
    Cut the original model from 'layer', and add a GlobalAveragePooling2D() for storage.

    :param layer1: where to cut the model
    :param layer2: where to cut the model
    :param layer3: where to cut the model

    :return: A Xception model cut by 'name'
    """
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))

    # output1
    x1 = base_model.layers[layer1].output
    # add a global spatial average pooling layer
    x1 = GlobalAveragePooling2D()(x1)

    # output2
    x2 = base_model.layers[layer2].output
    x2 = GlobalAveragePooling2D()(x2)

    # output3
    x3 = base_model.layers[layer3].output
    x3 = GlobalAveragePooling2D()(x3)

    # return extractor
    extractor_model = Model(inputs=base_model.input, outputs=[x1, x2, x3])
    return extractor_model


def extractor_by_layer(layer):
    """Bottleneck feature extractor

    Extract bottleneck feature used for fine-tuning last several layers.
    Cut the original model from 'layer', and add a GlobalAveragePooling2D() for storage.

    :param layer: where to cut the model

    :return: A Xception model cut by 'layer'
    """
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))
    # add a global spatial average pooling layer
    x = base_model.layers[layer].output
    x = GlobalAveragePooling2D()(x)

    # return extractor
    extractor_model = Model(inputs=base_model.input, outputs=x)
    return extractor_model


def extractor_by_name(name):
    """Bottleneck feature extractor

    Extract bottleneck feature used for fine-tuning last several layers.
    Cut the original model from 'layer', and add a GlobalAveragePooling2D() for storage.

    :param name: where to cut the model

    :return: A Xception model cut by 'name'
    """
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))
    # add a global spatial average pooling layer
    x = base_model.get_layer(name).output
    x = GlobalAveragePooling2D()(x)

    # return extractor
    extractor_model = Model(inputs=base_model.input, outputs=x)
    return extractor_model


def branch(inputs, num_filters, num_labels, name='b', activation='relu', use_softmax=True):
    """Xception type branch

    A Xception type branch, use for exploiting hierarchical labels.

    :param inputs: input tensor
    :param num_filters: number of filters used in each conv layer
    :param num_labels: output size
    :param name: branch name
    :param activation: activation function
    :param use_softmax: whether or not apply softmax to logits

    :return: a branch of model
    """
    # input = MaxPooling2D((3, 3), strides=(2, 2), name=name+'_max_pool')(input)
    b = SeparableConv2D(num_filters, (3, 3), padding='same', use_bias=False, name=name+'_sepconv1')(inputs)
    b = BatchNormalization(name=name+'_sepconv1_bn')(b)
    b = Activation(activation, name=name+'_sepconv1_act')(b)
    b = SeparableConv2D(num_filters, (3, 3), padding='same', use_bias=False, name=name+'_sepconv2')(b)
    b = BatchNormalization(name=name+'_sepconv2_bn')(b)
    b = Activation(activation, name=name+'_sepconv2_act')(b)
    b = GlobalAveragePooling2D(name=name+'_avg_pool')(b)

    if use_softmax:
        output = Dense(num_labels, activation='softmax', name=name)(b)
    else:
        # output (no softmax for DARC1)
        output = Dense(num_labels, name=name)(b)
    return output


def xception_branch(num_dense):
    """Xception model with branch

    :param num_dense:
    :return:
    """
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


def base_model_3outputs(layer1, layer2, layer3):
    """Base model for further training

    :param layer1:
    :param layer2:
    :param layer3:

    :return: input, output1, output2, output3
    """
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))

    # output1
    x1 = base_model.layers[layer1].output
    # add a global spatial average pooling layer
    x1 = GlobalAveragePooling2D()(x1)

    # output2
    x2 = base_model.layers[layer2].output
    x2 = GlobalAveragePooling2D()(x2)

    # output3
    x3 = base_model.layers[layer3].output
    x3 = GlobalAveragePooling2D()(x3)

    return base_model.input, x1, x2, x3


def model_last_block(input_shape, num_labels, num_units, use_softmax=True):
    """Random initialized layers

    :param input_shape:
    :param num_labels:
    :param num_units:
    :param use_softmax:
    :return:
    """
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
    x = Dense(num_units)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # add a output layer
    if not use_softmax:
        output = Dense(num_labels)(x)
    else:
        output = Dense(num_labels, activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=inputs, outputs=output)
    return model


def combine_model_3branch(num_units1, num_units2, num_units3, path_b1, path_b2, path_b3, use_softmax=True):
    """Combining whole model (3 branches)

    :param num_units1:
    :param num_units2:
    :param num_units3:
    :param path_b1:
    :param path_b2:
    :param path_b3:
    :return:
    """
    inputs, x1, x2, x3 = base_model_3outputs(95, 115, 131)

    # Load last layers after fine-tuning

    # branch1
    model_b1 = model_last_block(728, num_classes1, num_units=num_units1, use_softmax=use_softmax)
    model_b1.load_weights(path_b1)
    b1 = model_b1(x1)

    # branch2
    model_b2 = model_last_block(728, num_classes2, num_units=num_units2, use_softmax=use_softmax)
    model_b2.load_weights(path_b2)
    b2 = model_b2(x2)

    # main branch
    model_b3 = model_last_block(2048, num_classes, num_units=num_units3, use_softmax=use_softmax)
    model_b3.load_weights(path_b3)
    b3 = model_b3(x3)

    model = Model(inputs=inputs, outputs=[b1, b2, b3])
    return model


def combine_model(num_units, path_branch, use_softmax=True):
    """Combining whole model

    :param num_units:
    :param path_branch:
    :param use_softmax:
    :return:
    """

    base_model = Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))

    # Load last layers after fine-tuning

    # main branch
    model_b = model_last_block(2048, num_classes, num_units=num_units, use_softmax=use_softmax)
    model_b.load_weights(path_branch)
    output = model_b(base_model.output)

    model = Model(inputs=base_model.input, outputs=output)
    return model


def xception_3branch(num_units1, num_units2, num_units3, use_softmax=True):
    inputs, x1, x2, x3 = base_model_3outputs(95, 115, 131)

    # Load last layers after fine-tuning

    # branch1
    model_b1 = model_last_block(728, num_classes1, num_units=num_units1, use_softmax=use_softmax)
    b1 = model_b1(x1)

    # branch2
    model_b2 = model_last_block(728, num_classes2, num_units=num_units2, use_softmax=use_softmax)
    b2 = model_b2(x2)

    # main branch
    model_b3 = model_last_block(2048, num_classes, num_units=num_units3, use_softmax=use_softmax)
    b3 = model_b3(x3)

    model = Model(inputs=inputs, outputs=[b1, b2, b3])
    return model


def xception_3branch_test(num_units1, num_units2, num_units3, use_softmax=True):
    inputs, x1, x2, x3 = base_model_3outputs(95, 115, 131)

    # branch1
    b1 = Dense(num_units1)(x1)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    # add a output layer
    if not use_softmax:
        out1 = Dense(num_classes1)(b1)
    else:
        out1 = Dense(num_classes1, activation='softmax')(b1)

    # branch2
    b2 = Dense(num_units2)(x2)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    # add a output layer
    if not use_softmax:
        out2 = Dense(num_classes2)(b2)
    else:
        out2 = Dense(num_classes2, activation='softmax')(b2)

    # main branch
    b3 = Dense(num_units3)(x3)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)
    # add a output layer
    if not use_softmax:
        output = Dense(num_classes)(b3)
    else:
        output = Dense(num_classes, activation='softmax')(b3)

    model = Model(inputs=inputs, outputs=[out1, out2, output])
    return model
