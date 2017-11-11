import os
import io
from tqdm import *

import numpy as np
import pandas as pd
import bson

from model import DARC1

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import utils.utils as u


# Make weighted prediction for each product in test dataset
def make_weighted_prediction(test_data, test_gen, num_products, model_names, weights):
    model = []
    for i in range(len(model_names)):
        model.append(load_model(model_names[i]))
        #model[i] = load_model(model_names[i], custom_objects={'DARC1': DARC1})
    pred_cat_id = []

    with tqdm(total=num_products) as pbar:
        for count, product in enumerate(test_data):

            # process data

            num_imgs = len(product["imgs"])
            batch_x = np.zeros((num_imgs, 180, 180, 3), dtype='float32')
            for i in range(num_imgs):
                bson_img = product["imgs"][i]["picture"]
                # Load and preprocess the image.
                img = load_img(io.BytesIO(bson_img), target_size=(180, 180))
                x = img_to_array(img)
                x = test_gen.random_transform(x)
                x = test_gen.standardize(x)
                # Add the image to the batch.
                batch_x[i] = x

            # make predictions

            weighted_prediction = np.zeros((num_imgs, num_classes), dtype='float32')
            #for weight, name in zip(weights, model_names):
            for i in range(len(weights)):
                #model.load_weights(name)
                prediction = model[i].predict(batch_x, batch_size=num_imgs)
                weighted_prediction += weights[i] * prediction
            # predict product idx by the average of each images
            avg_pred = weighted_prediction.mean(axis=0)
            cat_idx = np.argmax(avg_pred)
            # Use idx2cat[] to convert the predicted category index back to the original class label.
            pred_cat_id.append(idx2cat[cat_idx])
            pbar.update()

    return pred_cat_id


# Input data files are available in the "../input/" directory.
data_dir = u.data_dir
utils_dir = u.utils_dir
model_dir = u.model_dir
result_dir = u.result_dir

test_bson_path = os.path.join(data_dir, "test.bson")
num_classes = u.num_classes
num_test_products = u.num_test_products

# # Part 4: Predicting

# First load the lookup tables from the CSV files.

categories_df = pd.read_csv(utils_dir+"categories.csv", index_col=0)
_, idx2cat = u.make_category_tables(categories_df)

val_images_df = pd.read_csv(utils_dir+"val_images.csv", index_col=0)
val_df = pd.read_pickle(utils_dir+"val_dataset.pkl")

submission_df = pd.read_csv(data_dir + "sample_submission.csv")

# Use BSONIterator to load the test set images in batches.
test_bson_file = open(test_bson_path, "rb")

test_datagen = ImageDataGenerator(
#    rescale=1./255
)
test_dataflow = bson.decode_file_iter(test_bson_file)


# model settings
dense_num = 128
model_prefix = 'Xception-pretrained-%d' % dense_num
nb_snapshots = 3
models_filenames = [model_dir+model_prefix+"-Best.h5"]
models_filenames.extend([model_dir+model_prefix+"-%d.h5" % (i+1) for i in range(nb_snapshots)])
#models_filenames = [model_dir+"Xception-nofc-pretrained-128.h5",model_dir+"Xception-nofc-pretrained-1282.h5",model_dir+"Xception-nofc-pretrained-1283.h5"]
# model weights
weights = [1. / len(models_filenames)] * len(models_filenames)

# make weighted predictions
pred_cat_id = make_weighted_prediction(test_dataflow, test_datagen, num_test_products, models_filenames, weights)

# generate submission
pred_cat_id_df = pd.DataFrame(pred_cat_id, columns=["category_id"], dtype=int)
pred_cat_id_df = pd.concat([submission_df["_id"], pred_cat_id_df], axis=1)
pred_cat_id_df.to_csv(result_dir+"submission.csv.gz", compression="gzip", index=False)
