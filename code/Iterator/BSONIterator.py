import io
import numpy as np
import bson

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import Iterator

import keras.backend as K

class BSONIterator(Iterator):
    def __init__(self, bson_file, image_table, product_table, num_label,
                 image_data_generator, num_label_level1=None, num_label_level2=None,
                 image_size=(180, 180), labelled=True, batch_size=32,
                 shuffle=False, seed=None, use_hierarchical_label=False):
        """
        :param bson_file:               The original bson file instance
        :param image_table:             The table that stores information of each image
        :param product_table:           The table that stores position of each product
        :param num_label:               The number of categories
        :param image_data_generator:    Data augmentation generator
        :param num_label_level1:        The number of categories in level1
        :param num_label_level2:        The number of categories in level2
        :param image_size:              The size of image
        :param labelled:                Check if the image is of validation or training set
        :param batch_size:              Size of the batch
        :param shuffle:                 Whether shuffle dataset or not
        :param seed:                    Random seed
        :param use_hierarchical_label:  Whether use hierarchical label or not
        """
        # Parameter initialization
        self.__file = bson_file
        self.__image_table = image_table
        self.__product_table = product_table
        self.__labelled = labelled
        self.__num_images = len(image_table)
        self.__num_label = num_label
        self.__num_label_level1 = num_label_level1
        self.__num_label_level2 = num_label_level2
        self.__image_data_generator = image_data_generator
        self.__image_size = tuple(image_size)
        self.__image_shape = self.__image_size + (3,)
        self.__use_hierarchical_label = use_hierarchical_label
        # Pass parameter back to super class
        super(BSONIterator, self).__init__(self.__num_images, batch_size, shuffle, seed)

        # Print out information
        print("Found %d images." % self.__num_images)

    def _get_batches_of_transformed_samples(self, index_array):
        """
        :param index_array:
        :return: The batch of samples(pair of image and label)
        """

        # Initialize the zeros arrays to store image and label
        batch_x = np.zeros((len(index_array),) + self.__image_shape, dtype=K.floatx())
        if self.__labelled:
            batch_y = np.zeros((len(batch_x), self.__num_label), dtype=K.floatx())
            if self.__use_hierarchical_label:
                batch_y_level1 = np.zeros((len(batch_x), self.__num_label_level1), dtype=K.floatx())
                batch_y_level2 = np.zeros((len(batch_x), self.__num_label_level2), dtype=K.floatx())

        # Generate batch
        for i, j in enumerate(index_array):
            # Protect file and data frame access with a lock.
            with self.lock:
                image_row = self.__image_table.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.__product_table.loc[product_id]

                # Read this product's data from the BSON file.
                self.__file.seek(offset_row["offset"])
                item_data = self.__file.read(offset_row["length"])

            # Grab the image from the product.
            item = bson.BSON.decode(item_data)
            img_idx = image_row["img_idx"]
            bson_img = item["imgs"][img_idx]["picture"]

            # Pre process the image.
            img = load_img(io.BytesIO(bson_img), target_size=self.__image_size)
            x = img_to_array(img)
            x = self.__image_data_generator.random_transform(x)
            x = self.__image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.__labelled:
                batch_y[i, image_row["category_idx"]] = 1
                if self.__use_hierarchical_label:
                    batch_y_level1[i, image_row["category_idx_level1"]] = 1
                    batch_y_level2[i, image_row["category_idx_level2"]] = 1

        if self.__labelled:
            if not self.__use_hierarchical_label:
                return batch_x, batch_y
            else:
                return batch_x, [batch_y_level1, batch_y_level2, batch_y]
        else:
            return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)