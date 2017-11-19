import io
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import Iterator

import keras.backend as K

class PickleGenerator(Iterator):
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
        super(PickleGenerator, self).__init__(self.__num_images, batch_size, shuffle, seed)
        # Print out information
        print("Found %d images." % self.__num_images)

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
