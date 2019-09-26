import foolbox
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import glob
import os
from absl import logging

logging._warn_preinit_stderr = 0

NUM_PARALLEL_EXEC_UNITS = 4
os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_AFFINITY"] = "none"

HEIGHT, WIDTH = 224, 224
current_directory = os.getcwd()
images_path = os.path.dirname(current_directory) + r"/images/100_black_swan/*.jpg"


def generate_adversarial_example(_attack, _image, _label):
    """
    Generates an adversarial example using the given _attack which has been previously implemented in foolbox

    :param _attack:      the _attack that will be used for generating the adversarial example
    :param _image:       the image that will be used for generating the adversarial example
    :param _label:       the _label of the class associated with the given image
    :return:            an image that is the adversarial example
    """
    adversarial = _attack(_image[:, :, ::-1], _label)
    # if the attack fails, adversarial will be None and a warning will be printed
    return adversarial


def image_getter(path):
    """
    Given a path to a image folder along with an specified extension, it reads al the images that fits the extension
    an puts them into a list

    :param path:        the path to folder along with the image extension
    :return:            a list of images
    """
    image_list = []
    for filename in glob.glob(path):
        im = image.load_img(filename, target_size=(WIDTH, HEIGHT))
        image_list.append(image.img_to_array(im))
    return image_list


def instantiate_resnet50():
    """
    Instantiates the convolutional model ResNet50 setting a pre processing of the images that'll be passed

    :return:        the model resnet50
    """
    keras.backend.set_learning_phase(0)
    kmodel = ResNet50(weights='imagenet')
    fmodel = foolbox.models.KerasModel(kmodel, bounds=(-255, 255))
    return fmodel


def plot_image_comparison(image_, adversarial):
    """
    Plots an image_, its given adversarial generated example and the difference between them in order to visualize
    the similarities

    :param image_:           the image
    :param adversarial:     the adversarial example of the image_, previously generated
    """
    if image_ is not None:
        plt.figure()

        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(image_ / 255)  # division by 255 to convert [0, 255] to [0, 1]
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Adversarial')
        plt.imshow(adversarial / 255)  # ::-1 to convert BGR to RGB
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Difference')
        difference = adversarial - image_
        plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
        plt.axis('off')

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


def adversarial_examples_comparison(img_path, _label, attack="fgsm"):
    """
    Given a set of images that belong to the class _label of the image dataset, adversarial examples
    are created using the Fast Gradient Sign Method over a resnet50 model

    :param attack:          the attack that will be used to create the adversarial examples
    :param img_path:        the path to the folder where the images are
    :param _label:          the label of the images' class
    """
    _images = image_getter(img_path)
    _images = [preprocess_input(img.copy()) for img in _images]

    # the resnet50 model is instantiated
    _model = instantiate_resnet50()
    if attack == "fgs":
        _attack = foolbox.attacks.GradientSignAttack(_model)
    elif attack == "fg":
        _attack = foolbox.attacks.GradientAttack(_model)
    elif attack == "fgsm":
        _attack = foolbox.attacks.FGSM(_model)
    elif attack == "dfl2":
        _attack = foolbox.attacks.DeepFoolL2Attack(_model)
    else:
        return -1

    # generation of adversarial examples
    _adversarial_examples = [generate_adversarial_example(_attack, img, _label) for img in _images]

    # comparison of both images
    for img, adv in zip(_images, _adversarial_examples):
        plot_image_comparison(restore_original_image_from_array(img.copy()),
                              restore_original_image_from_array(adv.copy()[:, :, ::-1]))
        print(decode_predictions(np.reshape([_model.predictions(img)], (-1, 1000))))
        print(decode_predictions(np.reshape([_model.predictions(adv)], (-1, 1000))))


if __name__ == "__main__":
    adversarial_examples_comparison(images_path, 100, "fgs")
