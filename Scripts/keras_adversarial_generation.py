from __future__ import absolute_import, division, print_function, unicode_literals

import random
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from imagenetData import ImageNetData
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.backend import clear_session

tf.compat.v1.enable_eager_execution()

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def define_model(num_classes):
    # Defining the model
    the_model = models.Sequential()
    the_model.add(layers.Conv2D(16, (7, 7), activation='relu', padding='same', input_shape=(224, 224, 3)))
    # model.add(layers.BatchNormalization())
    the_model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(dropout_rate))
    the_model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    # model.add(layers.BatchNormalization())
    the_model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(dropout_rate))
    the_model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(layers.BatchNormalization())
    the_model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(dropout_rate))
    the_model.add(layers.Conv2D(16, (7, 7), activation='relu', padding='same'))
    # model.add(layers.BatchNormalization())
    the_model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(dropout_rate))
    the_model.add(layers.Conv2D(num_classes, (1, 1)))
    the_model.add(layers.GlobalAveragePooling2D())

    the_model.summary()

    the_model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    return the_model


# Useful training functions
def validate(model):
    imageNet.shuffle_validation()
    batches = imageNet.get_validation_set(as_batches=True)
    accs = []
    xent_vals = []
    for batch in batches:
        data, labels = batch
        xentropy_val, acc = model.test_on_batch(data, y=labels,
                                                sample_weight=None, reset_metrics=True)
        accs.append(acc)
        xent_vals.append(xentropy_val)
    mean_xent = np.array(xent_vals).mean()
    mean_acc = np.array(accs).mean()
    return mean_acc, mean_xent


def to_test_model(model):
    batches = imageNet.get_test_set(as_batches=True)
    accs = []
    for batch in batches:
        data, labels = batch
        _, acc = model.test_on_batch(data, y=labels,
                                     sample_weight=None, reset_metrics=True)
        accs.append(acc)
    mean_acc = np.array(accs).mean()
    return mean_acc


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
    batch_number = int(images_per_class / batch_size)
    dropout_rate = .2
    classes = [96, 950]  # ,447, 530, 592, 950, 96]
    n_classes = len(classes)
    imageNet = ImageNetData(classes, images_per_class=500,
                            batch_size=batch_size, validation_proportion=0.4, augment_data=True)

    model = define_model(n_classes)

    epochs = 5
    history = {'loss': [], 'accuracy': []}

    imageNet.reset()
    prev_epoch = imageNet.get_epoch()

    t_i = time.time()
    n_batches = imageNet.n_batches
    val_acc_vals = []
    test_acc_vals = []
    inference_time = []

    while imageNet.get_epoch() < epochs:
        epoch = imageNet.get_epoch()

        batch, batch_idx = imageNet.next_batch()
        batch_data = batch[0].astype(float)
        batch_labels = batch[1].astype(float)

        step = batch_idx + epoch * n_batches

        # gradient (by layer) statistics over last training batch & validation summary
        if batch_idx == 0:
            loss, acc = model.test_on_batch(batch_data, y=batch_labels,
                                            sample_weight=None, reset_metrics=True)

            validation_accuracy, validation_loss = validate(model)
            print('[Epoch %d, it %d] Training acc. %.3f, loss %.3f. \ Valid. acc. %.3f, loss %.3f' % (
                epoch,
                step,
                acc,
                loss,
                validation_accuracy,
                validation_loss
            ))
            val_acc_vals.append(validation_accuracy)
            test_accuracy = to_test_model(model)
            test_acc_vals.append(test_accuracy)
            print("Time elapsed %.2f minutes" % ((time.time() - t_i) / 60.0))

        model_metrics = model.train_on_batch(batch_data, y=batch_labels,
                                             sample_weight=None, class_weight=None,
                                             reset_metrics=True)

        history['loss'].append(model_metrics[0])
        history['accuracy'].append(model_metrics[1])

    val_acc_vals = np.array(val_acc_vals)
    test_acc_vals = np.array(test_acc_vals)
    best_epoch = np.argmax(val_acc_vals)
    test_acc_at_best = test_acc_vals[best_epoch]
    print('*' * 30)
    print("Testing set accuracy @ epoch %d (best validation acc): %.4f" % (best_epoch, test_acc_at_best))
    print('*' * 30)

    """
    x_adversarial_test, y_adversarial_test = next(generate_adversarials(10))
    print("Base accuracy on adversarial images:", model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))
    """

    """
    res = []
    lol = np.linspace(0, 3, num=30)
    for epsilon in tqdm(lol):
        x_adversarial_test, y_adversarial_test = next(generate_adversarials(epsilon, 10))
        res.append(model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0)[1])

    plt.plot(lol, res, '*')
    """

    epsilon = 8
    x_adversarial_test, y_adversarial_test = next(generate_adversarials(epsilon, 5))
    x_adversarial_train, y_adversarial_train = next(generate_adversarials(epsilon, 5))

    test_accuracy_before = to_test_model(model)
    adv_test_accu_before = model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0)[1]

    model.fit(x_adversarial_train, y_adversarial_train, batch_size=32, epochs=5)

    adv_test_accu_after = model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0)[1]
    test_accuracy_after = to_test_model(model)

    clear_session()
