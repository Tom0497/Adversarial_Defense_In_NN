import numpy as np
import tensorflow as tf
from imagenetData import labels_to_one_hot


def adversarial_pattern(model, image, label):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)

    gradient = tape.gradient(loss, image)

    signed_grad = tf.sign(gradient)

    return signed_grad


def adversarial_step_ll(model, num_classes, image):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        y_ll = model(image).numpy().argmin()
        y_ll = labels_to_one_hot([y_ll], num_classes)[0]
        loss = tf.keras.losses.MSE(y_ll, prediction)

    gradient = tape.gradient(loss, image)

    signed_grad = -1 * tf.sign(gradient)

    return signed_grad


def generate_adversarial(model, examples, labels, num_classes, number_of_examples=None, image_list=None, epsilon=None,
                         use_step_ll=False):
    while True:
        x = []
        original_x = []
        y = []

        if image_list is None:
            image_list = list(range(len(labels)))
            np.random.shuffle(image_list)

        if number_of_examples is None:
            number_of_examples = len(image_list)

        for example in range(number_of_examples):
            n = image_list[example]
            original_x.append(n)

            label = labels[n]
            image = examples[n]

            if use_step_ll:
                perturbations = adversarial_step_ll(model, num_classes, image.reshape(1, 224, 224, 3)).numpy()
            else:
                perturbations = adversarial_pattern(model, image.reshape((1, 224, 224, 3)), label).numpy()

            if epsilon is None:
                epsilon = tf.abs(tf.random.truncated_normal([1, 1], mean=0, stddev=1)).numpy()[0][0]

            adversarial = image + perturbations * epsilon

            x.append(adversarial)
            y.append(labels[n])

        x = np.asarray(x).reshape((number_of_examples, 224, 224, 3))
        original_x = np.asarray(original_x)
        y = np.asarray(y)

        yield x, original_x, y