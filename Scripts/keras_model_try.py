from __future__ import absolute_import, division, print_function, unicode_literals

import random

import models_and_utils as mm
import numpy as np
import tensorflow as tf
from imagenetData import ImageNetData
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.callbacks import ModelCheckpoint

tf.compat.v1.enable_eager_execution()

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def adversarial_pattern(image, label):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)

    gradient = tape.gradient(loss, image)

    signed_grad = tf.sign(gradient)

    return signed_grad


def generate_adversarials(epsilon, number_of_examples):
    while True:
        x = []
        y = []

        for example in range(number_of_examples):
            for b in range(batch_number):
                N = random.randint(0, batch_size - 1)

                batch, batch_idx = imageNet.next_batch()
                batch_data = batch[0].astype(float)
                batch_labels = batch[1].astype(float)

                label = batch_labels[N]
                image = batch_data[N]

                perturbations = adversarial_pattern(image.reshape((1, 224, 224, 3)), label).numpy()

                adversarial = image + perturbations * epsilon

                x.append(adversarial)
                y.append(batch_labels[N])

        x = np.asarray(x).reshape((number_of_examples * batch_number, 224, 224, 3))
        y = np.asarray(y)

        yield x, y


if __name__ == "__main__":
    # Load dataset
    batch_size = 64
    images_per_class = 500
    epochs = 30
    batch_number = int(images_per_class / batch_size)
    classes = [96, 950, 530]  # ,447, 530, 592, 950, 96]
    n_classes = len(classes)
    imageNet = ImageNetData(classes, images_per_class=500,
                            batch_size=batch_size, validation_proportion=0.4)

    model = mm.define_model(n_classes, own_num=1)

    checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    imageNet.reset()

    x_train, y_train = imageNet.get_train_set()
    x_val, y_val = imageNet.get_validation_set()
    x_test, y_test = imageNet.get_test_set()

    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
              callbacks=[checkpoint])

    print("Base accuracy in regular images : {}".format(model.evaluate(x=x_test, y=y_test, verbose=0)))

    clear_session()
