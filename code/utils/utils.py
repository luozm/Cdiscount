"""
Initial parameters, and common functions
"""

# Relative path of directories
data_dir = "../data/input/"
utils_dir = "../data/utils/"
log_dir = "../data/logs/"
model_dir = "../data/weights/"
result_dir = "../data/results/"

# Initial parameters for training
num_train_products = 7069896
num_test_products = 1768182
num_classes = 5270
num_class_level_one = 49
num_class_level_two = 483


# Create dictionaries for quick lookup of `category_id` to `category_idx` mapping.
def make_category_tables(category_table):
    """
    This function map the category id to label id
    Category id is a ten digits number, for example 1000005605
    Label id is a consecutive number from 0 to the number of labels - 1, for example 50000
    :param category_table:
    :return: Two table that map between category id and label id
    """
    category2label = {}
    label2category = {}
    for item in category_table.itertuples():
        category_id = item[0]
        label_id = item[4]
        category2label[category_id] = label_id
        label2category[label_id] = category_id
    return category2label, label2category
