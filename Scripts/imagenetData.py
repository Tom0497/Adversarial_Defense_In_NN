from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from zipfile import ZipFile

import pickle
import numpy as np
from sklearn.model_selection import train_test_split

use_colab = True

if use_colab:
    # for google collaboratory purposes
    DIR_BINARIES = r"/content/drive/My Drive/ImageNetDataSets/ImageNet_64/"
else:
    # for local use
    DIR_BINARIES = os.path.dirname(os.getcwd()) + r"/data/"


def unpickle(filename):
    """
    Reads a file that saves data in a pickle format the returns the data within it

    :param filename:        the path to the file that contains the data
    :return:                an array o data-frame that contains the data from the file
    """
    f = open(filename, 'rb')
    dic = pickle.load(f, encoding='latin1')
    f.close()
    return dic


def batch_to_bc01(batch, img_size):
    """
    Converts ImageNet sample to bc01 tensor

    :param img_size:    the size of the images contained in the batch
    :param batch:       the original data that's not ordered
    :return:            the data as a batch with dimensions (batch_size, channels, height, width)
    """
    return batch.reshape([-1, 3, img_size, img_size])


def batch_to_b01c(batch, img_size):
    """
    Converts ImageNet sample to b01c tensor

    :param img_size:    the size of the images contained in the batch
    :param batch:       the original data that's not ordered
    :return:            the data as a batch with dimensions (batch_size, height, width, channels)
    """
    return batch_to_bc01(batch, img_size).transpose(0, 2, 3, 1)


def labels_to_one_hot(labels):
    """
    Converts list of integers to numpy 2D array with one-hot encoding

    :param labels:      the labels associated with some data in a vector form
    :return:            one-hot encoding version of labels, therefore in a matrix
    """
    n = len(labels)
    one_hot_labels = np.zeros([n, 1000], dtype=int)
    one_hot_labels[np.arange(n), labels] = 1
    return one_hot_labels


class ImageNetData:
    """
    This class handles the use of ImageNet data set in one of its version, could be images of 8x8, 16x16, 32x32 or 64x64
    Besides handling the data loading, it provides methods for getting the data as batches, getting the epoch, and also
    allows to get the validation and test data
    """
    def __init__(self, batch_size=100, validation_proportion=0.1, augment_data=False, img_size=8):
        """
        It handles the creation of an instance of the ImageNetData class

        :param batch_size:              the wanted size of the batches of images
        :param validation_proportion:   the proportion of the training data used for validation
        :param augment_data:            bool to indicate if data augmentation will be use
        :param img_size:                the images' size, could be 8, 16, 32, 64
        """
        data_available = os.path.isfile(DIR_BINARIES+'train_data_batch_1')
        if not data_available:
            print('Unzipping files...')
            for item in os.listdir(DIR_BINARIES):           # loop through items in dir
                if item.endswith(".zip"):                   # check for ".zip" extension
                    file_name = DIR_BINARIES + "/" + item   # path for file to be extracted
                    zip_ref = ZipFile(file_name)            # create zipfile object
                    zip_ref.extractall(DIR_BINARIES)        # extract file to dir
                    zip_ref.close()                         # close file

        # Training set
        train_data_list = []
        self.train_labels = []
        for bi in range(1, 10):
            d = unpickle(DIR_BINARIES+'train_data_batch_'+str(bi))
            train_data_list.append(d['data'])
            self.train_labels += d['labels']
        self.train_labels = np.asarray(self.train_labels) - 1
        self.train_data = np.concatenate(train_data_list, axis=0).astype(np.float32)
        
        # Validation set
        assert 0. < validation_proportion < 1.
        self.train_data, self.validation_data, self.train_labels, self.validation_labels = train_test_split(
            self.train_data, self.train_labels, test_size=validation_proportion, random_state=1)
                
        # Test set
        d = unpickle(DIR_BINARIES+'val_data')
        self.test_data = d['data'].astype(np.float32)
        self.test_labels = np.asarray(d['labels']) - 1

        # Normalize data
        mean = self.train_data.mean(axis=0)
        std = self.train_data.std(axis=0)
        self.train_data = (self.train_data-mean)/std
        self.validation_data = (self.validation_data-mean)/std
        self.test_data = (self.test_data-mean)/std

        # Converting to b01c and one-hot encoding
        self.train_data = batch_to_b01c(self.train_data, img_size)
        self.validation_data = batch_to_b01c(self.validation_data, img_size)
        self.test_data = batch_to_b01c(self.test_data, img_size)
        self.train_labels = labels_to_one_hot(self.train_labels)
        self.validation_labels = labels_to_one_hot(self.validation_labels)
        self.test_labels = labels_to_one_hot(self.test_labels)

        np.random.seed(seed=1)
        self.augment_data = augment_data
            
        # Batching & epochs
        self.batch_size = batch_size
        self.n_batches = len(self.train_labels)//self.batch_size
        self.current_batch = 0
        self.current_epoch = 0
        
    def next_batch(self):
        """
        :return:    a tuple with batch and batch index
        """
        start_idx = self.current_batch*self.batch_size
        end_idx = start_idx + self.batch_size 
        batch_data = self.train_data[start_idx:end_idx]
        batch_labels = self.train_labels[start_idx:end_idx]
        batch_idx = self.current_batch

        if self.augment_data:
            if np.random.randint(0, 2) == 0:
                batch_data = batch_data[:, :, ::-1, :]
            batch_data += np.random.randn(self.batch_size, 1, 1, 3)*0.05
            
        # Update self.current_batch and self.current_epoch
        self.current_batch = (self.current_batch+1)%self.n_batches
        if self.current_batch != batch_idx+1:
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

    # TODO: refactor getTestSet and getValidationSet to avoid code replication
    def get_test_set(self, as_batches=False):
        """
        Get the test data from the dataset

        :param as_batches:       boolean to indicate if the data is wanted in batches
        :return:                the test data (images, labels)
        """
        if as_batches:
            batches = []
            for i in range(len(self.test_labels)//self.batch_size):
                start_idx = i*self.batch_size
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
            for i in range(len(self.validation_labels)//self.batch_size):
                start_idx = i*self.batch_size
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
    imageNet8 = ImageNetData(batch_size=64, img_size=8)
    batch, batch_idx = imageNet8.next_batch()
    print(batch_idx, imageNet8.n_batches, imageNet8.get_epoch())
    batches = imageNet8.get_test_set(as_batches=True)
    print(len(batches))
    data, labels = imageNet8.get_validation_set()
    print(labels.sum(axis=0))
    _, labels = imageNet8.get_test_set()
    print(labels.sum(axis=0))
