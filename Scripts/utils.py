import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score

HEIGHT, WIDTH = 224, 224
current_directory = os.getcwd()
images_path = os.path.dirname(current_directory) + r"/images/"


def image_getter(path):
    """
    Given a path to a image folder along with an specified extension, it reads al the images that fits the extension
    an puts them into a list

    :param path:        the path to folder along with the image extension
    :return:            a list of images
    """
    image_list = []
    image_name = []
    for filename in glob.glob(path):
        try:
            im = image.load_img(filename, target_size=(WIDTH, HEIGHT))
            image_list.append(image.img_to_array(im))
            image_name.append(filename)
        except IOError as e:
            print(e)
    return image_list


def plot_image_comparison(image_, adversarial, title_img="", title_adv=""):
    """
    Plots an image_, its given adversarial generated example and the difference between them in order to visualize
    the similarities

    :param title_adv:
    :param title_img:
    :param image_:           the image
    :param adversarial:     the adversarial example of the image_, previously generated
    """
    if image_ is not None:
        plt.figure()

        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(image_ / 255)  # division by 255 to convert [0, 255] to [0, 1]
        plt.axis('off')
        plt.title(title_img)

        plt.subplot(1, 3, 2)
        plt.title('Adversarial')
        plt.imshow(adversarial / 255)  # ::-1 to convert BGR to RGB
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Difference')
        difference = adversarial - image_
        plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
        plt.axis('off')
        plt.title(title_adv)

        plt.tight_layout()
        plt.show()


def restore_original_image_from_array(x, data_format=None):
    """
    It reverses the pre processing of the images that was needed before passing the images to the network

    :param x:               the image that will be processed
    :param data_format:     indicates if channels are the first dimension of the image array
    :return:                the processed image in its original format
    """
    mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return np.clip(x, 0, 255)


def one_hot(pred):
    """
    It encodes the input into a one-hot encoding, usually for classification tasks

    :param pred:        the predicted labels of a classification problem
    :return:            an array with one-hot encoding format
    """
    pred_classes = np.argmax(pred, 1)
    one_hots = np.zeros_like(pred)
    one_hots[np.arange(len(pred_classes)), pred_classes] = 1
    return one_hots


def get_accuracy(real_target, model_pred):
    """
    Gives the accuracy of a model based on the ideal and the predicted labels

    :param real_target:         the expected labels to be predicted
    :param model_pred:          the actual predicted labels from the model
    :return:                    the accuracy of the model
    """
    _y_pred = one_hot(model_pred)
    _y_real = np.zeros_like(_y_pred)
    _y_real[np.arange(_y_pred.shape[0]), real_target] = 1

    return accuracy_score(_y_real, _y_pred)


def get_accuracy_top5(real_target, model_pred):
    """
    Gives the accuracy of a model based on the ideal and the predicted labels, using the top-5 predictions
    this means that the real target can be any of the top-5 predictions of the model

    :param real_target:         the expected labels to be predicted
    :param model_pred:          the actual predicted labels from the model
    :return:                    the top-5 accuracy of the model
    """
    top_5 = (-model_pred).argsort(axis=-1)[:, :5]
    is_in_top5 = [real_target[i] if np.in1d(top_5[i, :], real_target[i]).sum()
                  else top_5[i, 0] for i in range(top_5.shape[0])]

    _y_pred = np.zeros_like(model_pred)
    _y_pred[np.arange(_y_pred.shape[0]), is_in_top5] = 1

    _y_real = np.zeros_like(_y_pred)
    _y_real[np.arange(_y_pred.shape[0]), real_target] = 1

    return accuracy_score(_y_real, _y_pred)
