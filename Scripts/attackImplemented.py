import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import foolbox

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import backend as k
from Scripts.foolbox_image_generator import image_getter, restore_original_image_from_array
from tensorflow.keras.utils import to_categorical


def fastGradientAttack(model, image, epsilon, alpha, type=None, preprocess=False):
    """
    It generates an adversarial example of an image by performing an adversarial attack over a model using one of three
    possible gradient based methods: Fast Gradient (FG), Fast Gradient Sign (FGS), Random Fast Gradient Sign (RFGS).

    :param model:       The model that's being attacked, which must have a computable gradient
    :param image:       The image that it'll serve as source for the adversarial example
    :param epsilon:     An hyper-parameter of the attack, it weights the gradient
    :param alpha:       An hyper-parameter for the RFGS attack, used as noise standard deviation for the image
    :param type:        Type of the attack, could be: 'fg'-->(FG), 'rfgs'-->(RFGS), any-->(FGS)
    :param preprocess:  Indicates if the image's been pre-processed before feeding to the network
    :return:            The generated adversarial example as an image with values between 0 and 1.
    """
    if preprocess:
        image = preprocess_input(image)

    y_predicted = model.predict(image.copy()[np.newaxis, :]).argmax()
    y_pred_one_hot = to_categorical(y_predicted, 1000)[np.newaxis, :]

    image = restore_original_image_from_array(image.copy()) / 255
    if type == 'rfgs':
        image += alpha * np.random.randn(image.shape[0], image.shape[1], image.shape[2])
        epsilon -= alpha

    image = image[np.newaxis, :]

    gradient = model.optimizer.get_gradients(model.total_loss, model.input)
    symbolic_vars = (model._feed_inputs + model._feed_targets)
    gradient_function = k.function(symbolic_vars, gradient)
    output_gradient = np.squeeze(gradient_function((image.copy(), y_pred_one_hot)))

    image = np.squeeze(image)

    if type == 'fg':
        perturbation = output_gradient / np.linalg.norm(output_gradient)
        epsilon *= (1.7 * image.shape[0])
    else:
        perturbation = np.sign(output_gradient)

    attack_result = image + epsilon * perturbation

    return np.clip(attack_result, 0, 1)


def deepFoolAttack(model, image, norm=2, preprocess=False):
    """
    It generates an adversarial example of the given image using the Deep Fool gradient based method.

    :param model:       The model that's being attacked, which must have a computable gradient
    :param image:       The image that it'll serve as source for the adversarial example
    :param norm:        The norm of the optimization problem behind Deep Fool, either 2 or np.inf
    :param preprocess:  Indicates if the image's been pre-processed before feeding to the network
    :return:            The generated adversarial example as an image with values between 0 and 1.
    """
    if preprocess:
        image = preprocess_input(image)
    if not (norm in [2, np.inf]):
        norm = 2

    y_predicted = model.predict(image.copy()[np.newaxis, :]).argmax()

    fool_model = foolbox.models.KerasModel(model, bounds=(-255, 255))
    attack = foolbox.attacks.DeepFoolAttack(fool_model)(image,
                                                        label=y_predicted,
                                                        p=norm)

    attack_result = restore_original_image_from_array(attack)/255

    return np.clip(attack_result, 0, 1)


if __name__ == "__main__":
    model = ResNet50(weights='imagenet')
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    current_directory = os.getcwd()
    images_path = os.path.dirname(current_directory) + r"/images/1_goldfish/*.jpg"

    images, _ = image_getter(images_path)
    epsilon = 10**-3

    for img in images:
        img_adversarial = fastGradientAttack(model,
                                             img.copy(),
                                             epsilon,
                                             epsilon*.5,
                                             type='rfgs',
                                             preprocess=True)

        img_adv = img_adversarial.copy()*255
        difference = img_adv.copy() - img.copy()

        plt.figure()
        plt.subplot(131)
        plt.imshow(img.copy()/255)
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(difference/abs(difference).max() * 0.2 + 0.5)
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(img_adv/255)
        plt.axis('off')

        plt.show()
