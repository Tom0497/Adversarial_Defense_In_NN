import numpy as np
import tensorflow as tf
from adversarial_generation import adversarial_pattern, adversarial_step_ll


def generate_adversarial(model,
                         examples,
                         labels,
                         num_classes,
                         examples_num=None,
                         image_list=None,
                         epsilon=None,
                         attack_type='fgsm'):
    """
    It creates adversarial examples from images and a model in order to access it weights and gradients. The number of
    classes for classification task must be specified. It handles four types of attack. FGSM, FGM, R-FGSM y Step Least
    Likely.

    :param model:               model of classification
    :param examples:            images to transfer to adversarial
    :param labels:              labels of these images
    :param num_classes:         number of possible classes on the output of the model
    :param examples_num:        if specified, this amount of examples are created, if not, all images are used
    :param image_list:          if specified, only images at these indexes are used
    :param epsilon:             parameter for intensity of attack
    :param attack_type:         type of attack
    :return:
    """

    while True:
        x = []
        original_x = []
        y = []

        if image_list is None:
            image_list = list(range(len(labels)))
            np.random.shuffle(image_list)

        if examples_num is None:
            examples_num = len(image_list)

        for example in range(examples_num):
            n = image_list[example]
            original_x.append(n)

            label = labels[n]
            image = examples[n]

            if epsilon is None:
                epsilon = tf.abs(tf.random.truncated_normal([1, 1], mean=1.5, stddev=0.75)).numpy()[0][0]

            if attack_type == 'step_ll':
                perturbations = adversarial_step_ll(model, image.reshape((1, 224, 224, 3)), num_classes).numpy()
                perturbations = np.array([pert / np.linalg.norm(pert) for pert in perturbations])

            elif attack_type == 'rfgs':
                alpha = epsilon / 2
                image += alpha * np.random.randn(*image.shape)
                epsilon -= alpha
                perturbations = adversarial_pattern(model, image.reshape((1, 224, 224, 3)), label).numpy()
                perturbations = np.array([pert / np.linalg.norm(pert) for pert in perturbations])

            elif attack_type == 'fg':
                perturbations = adversarial_pattern(model, image.reshape((1, 224, 224, 3)), label).numpy()
                perturbations = np.array([pert / np.linalg.norm(pert) for pert in perturbations])

            elif attack_type == 'fgsm':
                perturbations = tf.sign(adversarial_pattern(model, image.reshape((1, 224, 224, 3)), label)).numpy()
                perturbations = np.array([pert / np.linalg.norm(pert) for pert in perturbations])

            else:
                print('No attack seems to fit with the attack given')
                break

            img_adversarial = image + 388 * perturbations * epsilon  # 388 = (224*224*3)**0.5

            x.append(img_adversarial)
            y.append(label)

        x = np.asarray(x).reshape((examples_num, 224, 224, 3))
        original_x = np.asarray(original_x)
        y = np.asarray(y)

        yield x, original_x, y
