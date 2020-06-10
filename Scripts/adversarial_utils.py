import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Scripts.imagenet_data import labels_to_one_hot

NUM_CLASSES = 1000  # ILSVRC's number of classes


def adversarial_pattern(model, image, label):
    """
    Calculates the gradient of the 'model', at the position 'image' with respect to the image's true 'label'
    :param model:   ImageNet model
    :param image:   image in which to evaluate gradient
    :param label:   true label of image, one hot format
    :return:        calculated gradient
    """
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)[0]
        loss = loss_object(label, prediction)

    gradient = tape.gradient(loss, image)

    return gradient


def adversarial_step_ll(model, image, num_classes):
    """
    Calculates the gradient of the 'model', at the position 'image' with respect to the label with minimum probability.
    The attack One Step Least Likely Class uses this function.
    :param model:       ImageNet model
    :param image:       image in which to evaluate gradient
    :param num_classes: number of possible classes on the output of the model

    :return:            calculated gradient
    """
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        y_ll = model(image).numpy().argmin()
        y_ll = labels_to_one_hot([y_ll], num_classes)[0]
        loss = tf.keras.losses.MSE(y_ll, prediction)

    signed_gradient = tape.gradient(loss, image)

    return -1 * tf.sign(signed_gradient)


def generate_perturbations(model, image, label, epsilon, num_classes, attack_type):
    """
    Calculates the adversarial perturbation added of 'image' for four types of attack: FGSM, FGM, R-FGSM and Step Least
    Likely Class.

    :param model:               classification model
    :param image:               image for which to calculate perturbation
    :param label:               true label of the  image, in one hot format
    :param epsilon:             parameter for intensity of attack
    :param attack_type:         type of attack, possible options:
                                          - 'fgsm'    for FGSM
                                          - 'fg'      for FGM
                                          - 'rfgs'    for R-FGSM
                                          - 'step_ĺl' for Step Least Likely Class
    :param num_classes:         number of possible classes on the output of the model

    :return:                    adversarial perturbation for image, None if attack type isn't valid
    """
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
        perturbations = None
    return perturbations


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
    classes for classification task must be specified. It handles four types of attack: FGSM, FGM, R-FGSM and Step Least
    Likely Class.

    :param model:               classification model
    :param examples:            images to transfer to adversarial
    :param labels:              labels of these images, in one hot format
    :param num_classes:         number of possible classes on the output of the model
    :param examples_num:        if specified, this amount of examples are created, if not, all images are selected
    :param image_list:          if specified, only images at these indexes are selected
    :param epsilon:             parameter for intensity of attack
    :param attack_type:         type of attack, possible options:
                                          - 'fgsm'    for FGSM
                                          - 'fg'      for FGM
                                          - 'rfgs'    for R-FGSM
                                          - 'step_ĺl' for Step Least Likely Class

    :return:                    tuple of: - adversarial examples generated from selected original examples
                                          - selected original examples
                                          - selected original examples true labels, in one hot format
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

            perturbations = generate_perturbations(model, image, label, epsilon, attack_type, num_classes)

            if perturbations is None:
                print('No attack seems to fit with the attack given')
                break

            img_adversarial = image + 388 * perturbations * epsilon  # 388 = (224*224*3)**0.5

            x.append(img_adversarial)
            y.append(label)

        x = np.asarray(x).reshape((examples_num, 224, 224, 3))
        original_x = np.asarray(original_x)
        y = np.asarray(y)

        yield x, original_x, y


def simple_generate_adversarial(model, examples, labels, epsilon, attack_type):
    """
    Simplified 'generate_adversarial' function. Only returns generated adversarial examples.

    :param model:               classification model
    :param examples:            images to transfer to adversarial
    :param labels:              labels of these images, in one hot format
    :param epsilon:             parameter for intensity of attack
    :param attack_type:         type of attack, possible options:
                                          - 'fgsm'    for FGSM
                                          - 'fg'      for FGM
                                          - 'rfgs'    for R-FGSM
                                          - 'step_ĺl' for Step Least Likely Class

    :return:                    generated adversarial examples
    """
    while True:
        x = []

        for idx, image in enumerate(tqdm(examples)):
            label = labels[idx]

            perturbations = generate_perturbations(model, image, label, epsilon, attack_type, NUM_CLASSES)

            if perturbations is None:
                print('No attack seems to fit with the attack given')
                break

            img_adv = image + 388 * perturbations * epsilon  # 388 = (224*224*3)**0.5, normalization purposes
            x.append(img_adv)

        x = np.asarray(x).reshape((len(examples), 224, 224, 3))

        yield x
