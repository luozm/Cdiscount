"""
Utils
"""

data_dir = "../data/input/"
utils_dir = "../data/utils/"
log_dir = "../data/logs/"
model_dir = "../data/weights/"
result_dir = "../data/results/"

num_train_products = 7069896
num_test_products = 1768182
num_classes = 5270


# Create dictionaries for quick lookup of `category_id` to `category_idx` mapping.
def make_category_tables(categories_df):
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat