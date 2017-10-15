import os
import io
from tqdm import *

import numpy as np
import pandas as pd
import bson

from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import utils.utils as u


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

test_offsets_df = pd.read_csv(utils_dir+"test_offsets.csv", index_col=0)
test_images_df = pd.read_csv(utils_dir+"test_images.csv", index_col=0)

submission_df = pd.read_csv(data_dir + "sample_submission.csv")

# Use BSONIterator to load the test set images in batches.
test_bson_file = open(test_bson_path, "rb")

test_datagen = ImageDataGenerator(rescale=1./255)
test_data = bson.decode_file_iter(test_bson_file)


# Running model.predict_generator() gives a list of 3095080 predictions, one for each image.
#
# The indices of the predictions correspond to the indices in test_images_df.
# After making the predictions, you probably want to average the predictions for products that have multiple images.
#
# Use idx2cat[] to convert the predicted category index back to the original class label.

model = load_model(model_dir+"Xception-pretrained-128-Best.h5")

pred_cat_id = []

with tqdm(total=num_test_products) as pbar:

    for c, d in enumerate(test_data):
        num_imgs = len(d["imgs"])
        batch_x = np.zeros((num_imgs, 180, 180, 3), dtype=K.floatx())

        for i in range(num_imgs):
            bson_img = d["imgs"][i]["picture"]

            # Load and preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=(180, 180))
            x = img_to_array(img)
            x = test_datagen.random_transform(x)
            x = test_datagen.standardize(x)

            # Add the image to the batch.
            batch_x[i] = x

        prediction = model.predict(batch_x, batch_size=num_imgs)
        # predict product idx by the average of each images
        avg_pred = prediction.mean(axis=0)
        cat_idx = np.argmax(avg_pred)
        pred_cat_id.append(idx2cat[cat_idx])
        pbar.update()

# Only predict category by the first image of products
pred_cat_id_df = pd.DataFrame(pred_cat_id, columns=["category_id"], dtype=int)
pred_cat_id_df = pd.concat([submission_df["_id"], pred_cat_id_df], axis=1)
pred_cat_id_df.to_csv(result_dir+"submission.csv.gz", compression="gzip", index=False)
