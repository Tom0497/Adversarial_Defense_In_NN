import foolbox
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

IMAGE_PATH = '/images/*.jpg'
current_directory = os.getcwd()
images_path = os.path.dirname(current_directory) + r"/images/*.jpg"


def generate_adversarial_example(_attack, image, _label):
    """
    Generates an adversarial example using the given _attack which has been previously implemented in foolbox

    :param _attack:      the _attack that will be used for generating the adversarial example
    :param image:       the image that will be used for generating the adversarial example
    :param _label:       the _label of the class associated with the given image
    :return:            an image that is the adversarial example
    """
    # ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
    image = np.asarray(image)
    adversarial = _attack(image[:, :, ::-1], _label)
    # if the attack fails, adversarial will be None and a warning will be printed
    return adversarial


def pre_process_image(image, width=224, height=224):
    """
    Pre process an image by adjusting its height and width to a fix value
    assumes that the image is a PIL.Image object

    :param image:       the image to be processed
    :param width:       the desired output image width
    :param height:      the desired output image height
    :return:            the input image resized into the desire size
    """
    image = image.resize((width, height), Image.BICUBIC)
    return image


def image_getter(path):
    """
    Given a path to a image folder along with an specified extension, it reads al the images that fits the extension
    an puts them into a list

    :param path:        the path to folder along with the image extension
    :return:            a list of images
    """
    image_list = []
    for filename in glob.glob(path):
        im = Image.open(filename)
        image_list.append(im)
    return image_list


def instantiate_resnet50():
    """
    Instantiates the convolutional model ResNet50 setting a pre processing of the images that'll be passed

    :return:        the model resnet50
    """
    keras.backend.set_learning_phase(0)
    kmodel = ResNet50(weights='imagenet')
    preprocessing = (np.array([104, 116, 123]), 1)
    fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)
    return fmodel


def plot_image_comparison(image, adversarial):
    """
    Plots an image, its given adversarial generated example and the difference between them in order to visualize
    the similarities

    :param image:           the image
    :param adversarial:     the adversarial example of the image, previously generated
    """
    if image is not None:
        plt.figure()

        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Adversarial')
        plt.imshow(adversarial[:, :, ::-1] / 255)  # ::-1 to convert BGR to RGB
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Difference')
        difference = adversarial[:, :, ::-1] - image
        plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
        plt.axis('off')

        plt.show()


if __name__ == "__main__":
    images = image_getter(images_path)
    images = [pre_process_image(img) for img in images]
    model = instantiate_resnet50()
    attack = foolbox.attacks.FGSM(model)
    label = 2
    adversarial_images = [generate_adversarial_example(attack, img, label) for img in images]
    images = [np.asarray(img) for img in images]
    for img, adv in zip(images, adversarial_images):
        plot_image_comparison(img, adv)
