"""
Keras iterators for loading BSON and Pickle files.

"""
import io
import numpy as np
import bson

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import Iterator

import keras.backend as K


class BSONIterator(Iterator):
    """BSON file iterator

    It creates batches of images (and their one-hot labels) directly from the BSON file.
    It can be used with multiple workers.

    """
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

        if self.__num_label is None:
            # Print out information
            print("Found %d images." % self.__num_images)
        else:
            print("Found %d images belonging to %d classes." % (self.__num_images, self.__num_label))

    def _get_batches_of_transformed_samples(self, index_array):
        """
        :param index_array: Array that determine indexes of samples
        :return: A batch of samples
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
            x = random_crop(x, self.__image_size)           # random crop the image
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


class PickleIterator(Iterator):
    """Pickle file iterator

    Use for loading validation set.
    Almost the same as BSONIterator.

    """
    def __init__(self, num_label, pickle_file, image_data_generator, batch_size=32,
                 num_label_level1=None, num_label_level2=None, use_hierarchical_label=False,
                 image_size=(180, 180), labelled=True, shuffle=False, seed=None,):
        """
        :param pickle_file:             The original bson file instance
        :param num_label:               The amount of categories
        :param num_label_level1:        The number of categories in level1
        :param num_label_level2:        The number of categories in level2
        :param use_hierarchical_label:  Whether use hierarchical label or not
        :param image_data_generator:    Data augmentation generator
        :param image_size:              The size of image
        :param labelled:                Check if the image is of validation or training set
        :param batch_size:              Size of the batch
        :param shuffle:                 Whether use the shuffle strategy
        :param seed:                    Random seed
        """
        self.__pickle_file = pickle_file
        self.__batch_size = batch_size
        self.__labelled = labelled
        self.__num_label = num_label
        self.__num_label_level1 = num_label_level1
        self.__num_label_level2 = num_label_level2
        self.__image_data_generator = image_data_generator
        self.__image_size = tuple(image_size)
        self.__image_shape = self.__image_size + (3,)
        self.__num_images = len(pickle_file)
        self.__use_hierarchical_label = use_hierarchical_label

        # pass arguments to super class
        super(PickleIterator, self).__init__(self.__num_images, batch_size, shuffle, seed)

        if self.__num_label is None:
            # Print out information
            print("Found %d images." % self.__num_images)
        else:
            print("Found %d images belonging to %d classes." % (self.__num_images, self.__num_label))

    def next(self):
        index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        """
        :param index_array:
        :return: Batch of samples(pair of image and label)
        """
        batch_x = np.zeros((len(index_array),) + self.__image_shape, dtype=K.floatx())
        if self.__labelled:
            batch_y = np.zeros((len(batch_x), self.__num_label), dtype=K.floatx())
            if self.__use_hierarchical_label:
                batch_y_level1 = np.zeros((len(batch_x), self.__num_label_level1), dtype=K.floatx())
                batch_y_level2 = np.zeros((len(batch_x), self.__num_label_level2), dtype=K.floatx())

        for i, j in enumerate(index_array):
            sample = self.__pickle_file.iloc[j]

            # Pre process the image.
            img = load_img(io.BytesIO(sample["image"]), target_size=self.__image_size)
            x = img_to_array(img)
            x = center_crop(x, self.__image_size)           # center crop the image
            x = self.__image_data_generator.random_transform(x)
            x = self.__image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.__labelled:
                batch_y[i, sample["label"]] = 1
                if self.__use_hierarchical_label:
                    batch_y_level1[i, sample["label_level1"]] = 1
                    batch_y_level2[i, sample["label_level2"]] = 1

        if self.__labelled:
            if not self.__use_hierarchical_label:
                return batch_x, batch_y
            else:
                return batch_x, [batch_y_level1, batch_y_level2, batch_y]
        else:
            return batch_x


def random_crop(x, random_crop_size, sync_seed=None, rng=np.random):
    # np.random.seed(sync_seed)
    w, h = x.shape[0], x.shape[1]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    # print('w: {}, h: {}, rangew: {}, rangeh: {}'.format(w, h, rangew, rangeh))
    offsetw = 0 if rangew == 0 else rng.randint(rangew)
    offseth = 0 if rangeh == 0 else rng.randint(rangeh)
    return x[offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1], :]


def center_crop(x, center_crop_size):
    centerw, centerh = x.shape[1]//2, x.shape[2]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[:, centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh]

