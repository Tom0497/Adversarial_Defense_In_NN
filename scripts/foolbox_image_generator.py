import foolbox
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import glob
import os
from absl import logging
from sklearn.metrics import accuracy_score

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
    elif attack == "SPixelAttack":
        _attack = foolbox.attacks.SinglePixelAttack(_model)
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


# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]


def plot_im_with_confidence(_images, y_pred, im_num=0):
    _, _image_class, _class_confidence = get_imagenet_label(np.reshape(y_pred[im_num, :], (-1, 1000)))
    plt.figure()
    plt.imshow(restore_original_image_from_array(images[im_num].copy()) / 255)
    plt.title('{} : {:.2f}% Confidence'.format(_image_class, _class_confidence * 100))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":

    model = ResNet50(weights='imagenet')
    fmodelf = foolbox.models.KerasModel(model, bounds=(-255, 255))

    images = image_getter(images_path)
    images = np.stack([preprocess_input(img.copy()) for img in images])
    label = 100

    y_pred = model.predict(images)
    y_real = np.ones(len(images), dtype=int)*label

    accuracy = get_accuracy(y_real, y_pred)
    accuracy_top5 = get_accuracy_top5(y_real, y_pred)

    attack = foolbox.attacks.SinglePixelAttack(fmodelf)
    adversarial_img = np.stack([generate_adversarial_example(attack, img, label) for img in images])

    y_pred_adv = model.predict(adversarial_img)
    accuracy_adv = get_accuracy(y_real, y_pred_adv)

    for i in range(len(images)):
        plot_im_with_confidence(images, y_pred_adv, i)
