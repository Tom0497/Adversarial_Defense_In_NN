from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.preprocessing import image

from DataExtractor import *

WIDTH, HEIGHT = 224, 224
use_colab = False

CLASSES_FILE = "imagenet_class_index.json"
URLS_FILE = "fall11_urls.txt"

if use_colab:
    # for google collaboratory purposes
    DIR_BINARIES = r"/content/drive/My Drive/ImageNetDataSets/ImageNet/Images"
    DICT_FILE_PATH = os.path.join("/content/drive/My Drive/ImageNetDataSets/ImageNet URLs and WnID", CLASSES_FILE)
    URLS_FILE_PATH = os.path.join("/content/drive/My Drive/ImageNetDataSets/ImageNet URLs and WnID", URLS_FILE)
    URLS_FOLDER_PATH = os.path.join("/content/drive/My Drive/ImageNetDataSets/ImageNet URLs and WnID", "/urls")
else:
    # for local use
    cwd = os.getcwd()
    directory_path = os.path.dirname(cwd)
    DIR_BINARIES = '../images'
    DICT_FILE_PATH = os.path.join("../image_metadata", CLASSES_FILE)
    URLS_FILE_PATH = os.path.join("../image_metadata", URLS_FILE)
    URLS_FOLDER_PATH = os.path.join("../image_metadata", "urls")


def get_sorted_dirs(path):
    """
    It receives a path to a folder that contains images separated by name in sub-folders,
    then sorts them by name in ascending order.

    :param path:    a path to a folder
    :return:        a python dictionary with (key, value) --> (class_num, class_folder_name)
    """
    classes_dirs = os.listdir(path)
    classes_nums = [int(x.partition("_")[0]) for x in classes_dirs]
    classes_dirs = [classes_dirs[i] for i in np.argsort(classes_nums)]
    classes_nums.sort()
    classes_dict = dict(zip(classes_nums, classes_dirs))
    return classes_dict


def image_getter(path, images_per_class, label):
    """
    Given a path to an image folder, it reads all the images in it an puts them into a list, if detects any
    kind of error when reading one, it deletes it.

    :param path:        			the path to the image folder
    :param images_per_class:        the number of images to bring to memory
    :param label:                   the label of the class in case more images are needed
    :return:            			a list of images
    """
    image_list = []
    filename_list = os.listdir(path)
    current_number_of_images = len(filename_list)
    if current_number_of_images < images_per_class:
        images_to_go = images_per_class - current_number_of_images
        download_images_by_int_label(label, DIR_BINARIES,
                                     URLS_FILE_PATH,
                                     URLS_FOLDER_PATH,
                                     DICT_FILE_PATH,
                                     download_limit=images_to_go,
                                     starting_url=current_number_of_images + 500)

    filename_list = os.listdir(path)

    for filename in filename_list:
        image_path = os.path.join(path, filename)
        try:
            im = image.load_img(image_path, target_size=(WIDTH, HEIGHT))
            image_list.append(image.img_to_array(im))
        except IOError as e:
            print(e)
            os.remove(image_path)

    return image_list


def unpickle(path, classes_labels_list, images_per_class):
    """
    Gets a certain number of images in a path specified by a folder path and an image class,
    with the possibility of specifying more that just one class.

    :param images_per_class:        it determines how many images in the given class bring to memory
    :param classes_labels_list:     a list which indicates the classes that'll be needed
    :param path:                    the path to the images folder
    :return:                        a python dictionary with both the images and the int label
    """

    classes_dirs = get_sorted_dirs(path)
    dic = {"data": [], "labels": []}
    for label in classes_labels_list:
        image_class_path = os.path.join(path, classes_dirs[label])
        images_in_class = image_getter(image_class_path, images_per_class, label)
        images_labels = [label for _ in range(len(images_in_class))]
        dic["data"] += images_in_class
        dic["labels"] += images_labels
    return dic


def labels_to_one_hot(original_labels, number_of_classes):
    """
    Converts list of integers to numpy 2D array with one-hot encoding

    :param number_of_classes:   how many classes are being classified
    :param original_labels:     the labels associated with some data in a vector form
    :return:                    one-hot encoding version of labels, therefore in a matrix
    """
    n = len(original_labels)
    one_hot_labels = np.zeros([n, number_of_classes], dtype=int)
    code_list = np.unique(original_labels)
    code_ix = np.argsort(code_list)
    code_dict = {code_list[i]: code_ix[i] for i in range(len(code_list))}
    labels_decoded = [code_dict[label] for label in original_labels]
    one_hot_labels[np.arange(n), np.asarray(labels_decoded)] = 1
    return one_hot_labels


class ImageNetData:
    """
    This class handles the use of ImageNet data set in one of its version, could be images of 8x8, 16x16, 32x32 or 64x64
    Besides handling the data loading, it provides methods for getting the data as batches, getting the epoch, and also
    allows to get the validation and test data.
    """

    def __init__(self, classes_list, images_per_class=200, batch_size=32, validation_proportion=0.1,
                 augment_data=False):
        """
        It handles the creation of an instance of the ImageNetData class

        :param batch_size:              the wanted size of the batches of images
        :param validation_proportion:   the proportion of the training data used for validation
        :param augment_data:            bool to indicate if data augmentation will be use
        """

        # Training set
        self.number_of_classes = len(classes_list)
        d = unpickle(DIR_BINARIES, classes_list, images_per_class)
        self.train_data = np.asarray(d['data']).astype(np.float32)
        self.train_labels = np.asarray(d['labels'])

        # Validation set
        assert 0. < validation_proportion < 1.
        self.train_data, self.validation_data, self.train_labels, self.validation_labels = train_test_split(
            self.train_data, self.train_labels, test_size=validation_proportion, random_state=1)

        # Test set
        self.validation_data, self.test_data, self.validation_labels, self.test_labels = train_test_split(
            self.validation_data, self.validation_labels, test_size=.5, random_state=1)

        # Normalize data
        self.mean = self.train_data.mean(axis=0)
        self.std = self.train_data.std(axis=0)
        self.train_data = (self.train_data - self.mean) / self.std
        self.validation_data = (self.validation_data - self.mean) / self.std
        self.test_data = (self.test_data - self.mean) / self.std

        # Converting to b01c and one-hot encoding
        self.train_labels = labels_to_one_hot(self.train_labels, self.number_of_classes)
        self.validation_labels = labels_to_one_hot(self.validation_labels, self.number_of_classes)
        self.test_labels = labels_to_one_hot(self.test_labels, self.number_of_classes)

        np.random.seed(seed=1)
        self.augment_data = augment_data

        # Batching & epochs
        self.batch_size = batch_size
        self.n_batches = len(self.train_labels) // self.batch_size
        self.current_batch = 0
        self.current_epoch = 0

    def next_batch(self):
        """
        :return:    a tuple with batch and batch index
        """
        start_idx = self.current_batch * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_data = self.train_data[start_idx:end_idx]
        batch_labels = self.train_labels[start_idx:end_idx]
        batch_idx = self.current_batch

        if self.augment_data:
            if np.random.randint(0, 2) == 0:
                batch_data = batch_data[:, :, ::-1, :]
            batch_data += np.random.randn(self.batch_size, 1, 1, 3) * 0.05

        # Update self.current_batch and self.current_epoch
        self.current_batch = (self.current_batch + 1) % self.n_batches
        if self.current_batch != batch_idx + 1:
            self.current_epoch += 1

            # shuffle training data
            new_order = np.random.permutation(np.arange(len(self.train_labels)))
            self.train_data = self.train_data[new_order]
            self.train_labels = self.train_labels[new_order]

        return (batch_data, batch_labels), batch_idx

    def get_epoch(self):
        """
        :return:    the current epoch of training data
        """
        return self.current_epoch

    def get_train_set(self):
        """
        Get the train data from the dataset

        :return:        the train data (images, labels)
        """
        return self.train_data, self.train_labels

    def get_test_set(self, as_batches=False):
        """
        Get the test data from the dataset

        :param as_batches:       boolean to indicate if the data is wanted in batches
        :return:                the test data (images, labels)
        """
        if as_batches:
            batches = []
            for i in range(len(self.test_labels) // self.batch_size):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_data = self.test_data[start_idx:end_idx]
                batch_labels = self.test_labels[start_idx:end_idx]

                batches.append((batch_data, batch_labels))
            return batches
        else:
            return self.test_data, self.test_labels

    def get_validation_set(self, as_batches=False):
        """
        Get the validation data from the dataset

        :param as_batches:       boolean to indicate if the data is wanted in batches
        :return:                the test data (images, labels)
        """
        if as_batches:
            batches = []
            for i in range(len(self.validation_labels) // self.batch_size):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_data = self.validation_data[start_idx:end_idx]
                batch_labels = self.validation_labels[start_idx:end_idx]

                batches.append((batch_data, batch_labels))
            return batches
        else:
            return self.validation_data, self.validation_labels

    def shuffle_validation(self):
        """
        It shuffles the validation data to give some randomness when use it
        """
        new_order = np.random.permutation(np.arange(len(self.validation_labels)))
        self.validation_labels = self.validation_labels[new_order]
        self.validation_data = self.validation_data[new_order]

    def reset(self):
        """
        Resets the counters of batch and epoch for the dataset
        """
        self.current_batch = 0
        self.current_epoch = 0


if __name__ == '__main__':
    classes = [592, 447]  # [950, 96, 530, 592, 447]
    imageNet8 = ImageNetData(classes, images_per_class=500,
                             batch_size=32,
                             validation_proportion=0.2,
                             augment_data=False)
    batch, example_batch_idx = imageNet8.next_batch()
    print(example_batch_idx, imageNet8.n_batches, imageNet8.get_epoch())
    examples_batches = imageNet8.get_test_set(as_batches=True)
    print(len(examples_batches))
    data, labels = imageNet8.get_validation_set()
    print(labels.sum(axis=0))
    _, labels = imageNet8.get_test_set()
    print(labels.sum(axis=0))
