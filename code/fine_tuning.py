"""
Fine-tuning last several layers for the model.

Zhimeng Luo
"""

import math
import random
import numpy as np
import pandas as pd

from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, TensorBoard, ModelCheckpoint

# custom callbacks, not the original keras one
from utils.callbacks import SnapshotModelCheckpoint
from utils import utils
from model.xception import model_last_block, combine_model, combine_model_3branch
from model.loss import sparse_darc1
from model.lr_schedule import exp_decay_schedule


data_dir = utils.data_dir
utils_dir = utils.utils_dir
log_dir = utils.log_dir
model_dir = utils.model_dir
result_dir = utils.result_dir

train_image_table = pd.read_csv(utils.utils_dir + "train_images.csv", index_col=0)
val_image_table = pd.read_csv(utils.utils_dir + "val_images.csv", index_col=0)

num_classes = [utils.num_class_level_one, utils.num_class_level_two, utils.num_classes]


def load_bottleneck(level):
    """Load bottleneck features and labels

    :param level: which level you want to load, 1,2, or 3

    :return: train_data, val_data, train_label, val_label
    """
    assert level == 1 or level == 2 or level == 3
    # Load bottleneck features
    train_data = np.load(utils_dir + 'bottleneck_features_train_level%d.npy' % level)
    val_data = np.load(utils_dir + 'bottleneck_features_val_level%d.npy' % level)
    # Load labels
    if level is 3:
        train_label = np.array(train_image_table)[:, 1]
        val_label = np.array(val_image_table)[:, 1]
    else:
        train_label = np.array(train_image_table)[:, level+1]
        val_label = np.array(val_image_table)[:, level+1]

    return train_data, val_data, train_label, val_label


# Random search for hyperparameter optimization
def train_random_search(level, max_val=10):
    result_list = []

    assert level == 1 or level == 2 or level == 3

    # Load data
    train_data, val_data, train_label, val_label = load_bottleneck(level)
    if level is 3:
        num_feature = 2048
    else:
        num_feature = 728

    for i in range(max_val):

        # ---------------------------------------------------------------------------------
        # Params to be validated
        #

#        initial_learning_rate = 10**random.uniform(-2, -4)
        initial_learning_rate = 0.003
        momentum = 1-10**random.uniform(0, -2)
#        momentum = 0.999
        decay_value = 0.3
        decay_epoch = 10

        alpha = 10**random.uniform(-3, -6)

#        batch_size = random.choice([64, 256, 1024])
        batch_size = 512
        num_final_dense_layer = random.choice([32, 128, 512])
#        num_final_dense_layer = 2048

        num_epoch = 5

        use_darc1 = True

        # ---------------------------------------------------------------------------------
        # Define model
        #

#        adam = Adam(lr=initial_learning_rate)
        sgd = SGD(lr=initial_learning_rate, momentum=momentum)

        if use_darc1:
            model = model_last_block(num_feature, num_classes[level-1], num_final_dense_layer, use_softmax=False)

            model.compile(
                optimizer=sgd,
                loss=sparse_darc1(alpha),
                metrics=["sparse_categorical_accuracy"],
            )

        else:
            model = model_last_block(num_feature, num_classes[level-1], num_final_dense_layer, use_softmax=True)

            model.compile(
                optimizer=sgd,
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"],
            )

        # Learning rate scheduler
#        lr_scheduler = LearningRateScheduler(exp_decay_schedule(initial_learning_rate, decay_value, decay_epoch))

        # ---------------------------------------------------------------------------------
        # Start training
        #

        model.fit(
            train_data,
            train_label,
            epochs=num_epoch,
            batch_size=batch_size,
            shuffle=True,
#            verbose=0,
#            callbacks=[lr_scheduler]
        )

        _, acc = model.evaluate(
            val_data,
            val_label,
            batch_size=1024,
            verbose=0)

        result_row = {
            "idx": i+1,
            "val_acc": acc,
            "alpha": alpha,
#            "batch": batch_size,
            "num_dense": num_final_dense_layer,
            "lr": initial_learning_rate,
            "mom": momentum}
        result_list.append(result_row)
        print(result_row)

    result_df = pd.DataFrame(result_list)
    result_df.to_csv(utils.result_dir + "result_sgd_darc1_level1.csv")


# Real train for fine tuning
def train_fine_tuning(level):
    assert level == 1 or level == 2 or level == 3

    # Load data
    train_data, val_data, train_label, val_label = load_bottleneck(level)
    if level is 3:
        num_feature = 2048
    else:
        num_feature = 728

    # Params
    initial_learning_rate = 0.05
    momentum = 0.65
    alpha = 3e-5
    decay_value = 0.1
    decay_epoch = 10
    batch_size = 512
    num_epoch = 30
    num_final_dense_layer = 2048
    model_prefix = 'Xception-last-layer-level%d-%d' % (level, num_final_dense_layer)
#    model_prefix = 'Xception-last-layer-nofc'

    model = model_last_block(num_feature, num_classes[level-1], num_final_dense_layer, False)
#    adam = Adam(lr=initial_learning_rate)
    sgd = SGD(lr=initial_learning_rate, momentum=momentum)

    model.compile(
        optimizer=sgd,
        loss=sparse_darc1(alpha),
#        loss='sparse_categorical_crossentropy',
        metrics=["sparse_categorical_accuracy"],
    )

    # Learning rate scheduler
#    lr_scheduler = LearningRateScheduler(exp_decay_schedule(initial_learning_rate, decay_value, decay_epoch))

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=decay_value, patience=1, verbose=1, min_lr=1e-5)

    # Visualization when training
    tensorboard = TensorBoard(
        log_dir=log_dir + "level%d/add_fc/%d/sgd/reduce1-0.1/" % (level, num_final_dense_layer),
        batch_size=batch_size,
        write_graph=False)

    check = ModelCheckpoint(
        model_dir + "level%d/reduce1-0.1/%s-{epoch:02d}-{val_sparse_categorical_accuracy:.3f}.h5" % (level, model_prefix),
        monitor="val_sparse_categorical_accuracy",
        save_best_only=False,
        save_weights_only=True,
    )

    history = model.fit(
        train_data,
        train_label,
        epochs=num_epoch,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(val_data, val_label),
        callbacks=[
#            lr_scheduler,
            check,
            tensorboard,
            reduce_lr,
        ]
    )
    model.save_weights(
        model_dir+'%s-%d-%.3f.h5' %
        (model_prefix, num_epoch, history.history['val_sparse_categorical_accuracy'][-1])
    )


def save_combine_model(num_units, num_epoch, score, sub_path=None, use_softmax=True):
    """Save combined model

    :param num_units:
    :param num_epoch:
    :param score:
    :param sub_path:
    :param use_softmax:
    """
    if sub_path is None:
        model_path = model_dir+'Xception-last-layer-%d-%d-%.3f.h5' % (num_units, num_epoch, score)
    else:
        model_path = model_dir+sub_path+'Xception-last-layer-%d-%d-%.3f.h5' % (num_units, num_epoch, score)
    model = combine_model(num_units, model_path, use_softmax=use_softmax)

    model_prefix = 'Xception-combine-%.3f' % score
    model.save_weights(model_dir+model_prefix+'.h5')


def save_combine_model_3branch(num_units1, num_units2, num_units3, path_b1, path_b2, path_b3, score, use_softmax=True):
    path_b1 = model_dir+"level1/"+path_b1
    path_b2 = model_dir + "level2/" + path_b2
    path_b3 = model_dir + "level3/" + path_b3

    model = combine_model_3branch(
        num_units1, num_units2, num_units3, path_b1, path_b2, path_b3, use_softmax)
    model_prefix = 'Xception-combine-3branch-%.3f' % score
    model.save_weights(model_dir+model_prefix+'.h5')


def calculate_val_times(num_solutions, top_k, confidence):
    """
    Calculate how many times need to be search,
    in order to achieve top_k result at that confidence.

    :param num_solutions: total number of points in the search space
    :param top_k: want to achieve top_k result
    :param confidence: confidence probability, between 0-1

    :return: search times
    """
    p = top_k/num_solutions
    times = math.ceil(math.log(1-confidence, 1-p))
    print("Need search: %d times." % times)
    return int(times)


#train_random_search(1, calculate_val_times(300, 10, 0.9))

#train_fine_tuning(3)

save_combine_model(2048, 25, 0.608)
"""
save_combine_model_3branch(
    512, 1024, 2048,
    path_b1="reduce1-0.1/Xception-last-layer-level1-512-26-0.724.h5",
    path_b2="reduce1-0.1/Xception-last-layer-level2-1024-18-0.656.h5",
    path_b3="reduce1-0.1/Xception-last-layer-level3-2048-24-0.600.h5",
    score=0.600,
    use_softmax=False,
)
"""
