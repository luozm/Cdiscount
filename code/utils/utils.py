"""
Initial parameters, and common functions
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# Relative path of directories
data_dir = "../data/input/"
utils_dir = "../data/utils/"
log_dir = "../data/logs/"
model_dir = "../data/weights/"
result_dir = "../data/results/"
parameter_dir = "./parameters/"

# Initial parameters for training
num_train_products = 7069896
num_test_products = 1768182
num_classes = 5270
num_class_level_one = 49
num_class_level_two = 483


# Create dictionaries for quick lookup of `category_id` to `category_idx` mapping.
def make_category_tables(category_table):
    """
    This function maps the category id to label id
    Category id is a ten digits number, for example 1000005605
    Label id is a consecutive number from 0 to the number of labels - 1, e.g. 5269
    :param category_table:
    :return: Two dict maps category id to label id
    """
    category2label = {}
    label2category = {}
    for item in category_table.itertuples():
        category_id = item[0]
        label_id = item[4]
        category2label[category_id] = label_id
        label2category[label_id] = category_id
    return category2label, label2category


def make_category_table_level1(category_level1_table, category_table):
    """
    This function maps the category id to level1 label id
    Category id is a ten digits number, for example 1000005605
    Level1 label id is a consecutive number from 0 to the number of level1 labels - 1, e.g. 48
    :param category_level1_table:
    :param category_table:
    :return: A dict maps category id to level1 label id
    """
    # Create a dict mapping 'category_level1_names' to 'category_level1_index'
    category_name2label_level1 = {}
    for item_level1 in category_level1_table.itertuples():
        category_name = item_level1[1]
        category_idx = item_level1[2]
        category_name2label_level1[category_name] = category_idx
    # Create a dict mapping 'category_id' to 'category_level1_index'
    category_id2label_level1 = {}
    for item in category_table.itertuples():
        category_id = item[0]
        category_idx = category_name2label_level1[item[1]]
        category_id2label_level1[category_id] = category_idx
    return category_id2label_level1


def make_category_table_level2(category_level2_table, category_table):
    """
    This function maps the category id to level2 label id
    Category id is a ten digits number, for example 1000005605
    Level2 label id is a consecutive number from 0 to the number of level2 labels - 1, e.g. 482
    :param category_level2_table:
    :param category_table:
    :return: A dict maps category id to level2 label id
    """
    # Create a dict mapping 'category_level1_names' to 'category_level1_index'
    category_name2label_level2 = {}
    for item_level2 in category_level2_table.itertuples():
        category_name = item_level2[1]
        category_idx = item_level2[2]
        category_name2label_level2[category_name] = category_idx
    # Create a dict mapping 'category_id' to 'category_level1_index'
    category_id2label_level2 = {}
    for item in category_table.itertuples():
        category_id = item[0]
        category_idx = category_name2label_level2[item[2]]
        category_id2label_level2[category_id] = category_idx
    return category_id2label_level2

# Get hyper parameters by reading json file
def get_hyper_parameter(model_name):
    json_file = parameter_dir + model_name + r".json"
    json_pointer = open(json_file, "r")

    hyper_parameter = json.load(json_pointer)

    return hyper_parameter

"""
# Show images after augmentation (use for debug)
count = 0
n = 4
while count <= 15:
    bx, by = next(train_gen)
    if count % n == 0:
        plt.figure(figsize=(14, 6))
    plt.subplot(1, n, count % n + 1)
    plt.imshow(bx[-1].astype(np.uint8))
    plt.axis('off')
    count += 1
plt.show()
"""

# visualizing losses and accuracy, and real learning rate
def visual_result(hist, lr=None):
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    xc = range(len(hist.history['loss']))

    # Losses
    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    # use bmh, classic,ggplot for big pictures
    plt.style.available
    plt.style.use(['classic'])
    plt.savefig('loss.jpg')

    # Accuracy
    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    # use bmh, classic,ggplot for big pictures
    plt.style.available
    plt.style.use(['classic'])
    plt.savefig('acc.jpg')

    # LearningRate
    if lr is not None:
        plt.figure(3, figsize=(7, 5))
        plt.plot(xc, lr)
        plt.xlabel('num of Epochs')
        plt.ylabel('learning rate')
        plt.title('Real Learning Rate')
        plt.grid(True)
        plt.legend(['lr'])
        plt.savefig('lr.jpg')

