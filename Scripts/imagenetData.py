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

"""
if use_colab:
    # for google collaboratory purposes
    DIR_BINARIES = r"/content/drive/My Drive/ImageNetDataSets/ImageNet/Images"
else:
    # for local use
    DIR_BINARIES = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "images")

classes_file = r"/imagenet_class_index.json"
urls_file = r"/fall11_urls.txt"
current_directory = os.getcwd()
dict_path_ = r"/content/drive/My Drive/ImageNetDataSets/ImageNet URLs and WnID" + classes_file
urls_path = r"/content/drive/My Drive/ImageNetDataSets/ImageNet URLs and WnID" + urls_file
images_path = r"/content/drive/My Drive/ImageNetDataSets/ImageNet" + r"/Images"
urls_folder_path = r"/content/drive/My Drive/ImageNetDataSets/ImageNet URLs and WnID" + r"/urls"
"""
DIR_BINARIES = r"/home/rai/Documentos/8vo Semestre/Inteligencia/Adversarial_Defense_In_NN/Scripts/images/"
classes_file = r"/imagenet_class_index.json"
urls_file = r"/fall11_urls.txt"
current_directory = os.getcwd()
dict_path_ = r"/home/rai/Documentos/8vo Semestre/Inteligencia/Adversarial_Defense_In_NN/image_metadata" + classes_file
urls_path = r"/home/rai/Documentos/8vo Semestre/Inteligencia/Adversarial_Defense_In_NN/image_metadata" + urls_file
images_path = r"/home/rai/Documentos/8vo Semestre/Inteligencia/Adversarial_Defense_In_NN/Scripts/images/"
urls_folder_path = r"/home/rai/Documentos/8vo Semestre/Inteligencia/Adversarial_Defense_In_NN/image_metadata/urls"


def get_sorted_dirs(path):
    classes_dirs = os.listdir(DIR_BINARIES)
    classes_nums = np.array([int(x.partition("_")[0]) for x in classes_dirs])
    classes_dirs = [classes_dirs[i] for i in np.argsort(classes_nums)]
    classes_nums.sort()
    classes_dict = dict(zip(classes_nums, classes_dirs))
    return classes_dict


def image_getter(path, images_per_class, label):
    """
    Given a path to a image folder along with an specified extension, it reads al the images that fits the extension
    an puts them into a list

    :param path:        			the path to folder along with the image extension
    :param images_per_class:
    :param label:
    :return:            			a list of images
    """
    image_list = []
    filename_list = os.listdir(path)
    current_number_of_images = len(filename_list)
    if current_number_of_images < images_per_class:
        images_to_go = images_per_class - current_number_of_images
        download_images_by_int_label(label, images_path,
                                     urls_path,
                                     urls_folder_path,
                                     dict_path_,
                                     download_limit=images_to_go,
                                     starting_url=current_number_of_images + 100)

    for filename in filename_list:
        try:
            im = image.load_img(path + r"/" + filename, target_size=(WIDTH, HEIGHT))
            image_list.append(image.img_to_array(im))
        except IOError as e:
            print(e)
    return image_list


def unpickle(path, classes_list, images_per_class):
    """
    Reads a file that saves data in a pickle format the returns the data within it

    :param filename:        the path to the file that contains the data
    :return:                an array o data-frame that contains the data from the file
    """

    classes_dirs = get_sorted_dirs(path)
    dic = {"data": [], "labels": []}
    for i in classes_list:
        the_path = path + r"/" + classes_dirs[i]
        label = int(classes_dirs[i].partition("_")[0])
        images = image_getter(the_path, images_per_class, label)
        labels = [label for j in range(len(images))]
        dic["data"] += images
        dic["labels"] += labels
    return dic


def labels_to_one_hot(labels, number_of_classes):
    """
    Converts list of integers to numpy 2D array with one-hot encoding

    :param labels:      the labels associated with some data in a vector form
    :return:            one-hot encoding version of labels, therefore in a matrix
    """
    n = len(labels)
    one_hot_labels = np.zeros([n, number_of_classes], dtype=int)
    code_list = np.unique(labels)
    code_ix = np.argsort(code_list)
    code_dict = {code_list[i]: code_ix[i] for i in range(len(code_list))}
    labels_decoded = [code_dict[label] for label in labels]
    one_hot_labels[np.arange(n), np.asarray(labels_decoded)] = 1
    return one_hot_labels


class ImageNetData:
    """
    This class handles the use of ImageNet data set in one of its version, could be images of 8x8, 16x16, 32x32 or 64x64
    Besides handling the data loading, it provides methods for getting the data as batches, getting the epoch, and also
    allows to get the validation and test data
    """

    def __init__(self, classes_list, images_per_class=200, batch_size=100, validation_proportion=0.1,
                 augment_data=False, img_size=224):
        """
        It handles the creation of an instance of the ImageNetData class

        :param batch_size:              the wanted size of the batches of images
        :param validation_proportion:   the proportion of the training data used for validation
        :param augment_data:            bool to indicate if data augmentation will be use
        :param img_size:                the images' size, could be 8, 16, 32, 64
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
    classes = [447, 530]  # 592, 950, 96]
    imageNet8 = ImageNetData(classes, images_per_class=200,
                             batch_size=32,
                             validation_proportion=0.2,
                             augment_data=False)
    batch, batch_idx = imageNet8.next_batch()
    print(batch_idx, imageNet8.n_batches, imageNet8.get_epoch())
    batches = imageNet8.get_test_set(as_batches=True)
    print(len(batches))
    data, labels = imageNet8.get_validation_set()
    print(labels.sum(axis=0))
    _, labels = imageNet8.get_test_set()
    print(labels.sum(axis=0))
