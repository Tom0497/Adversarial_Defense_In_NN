from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import time

from tensorflow.python.keras import layers, models, optimizers
from tensorflow.python.keras.applications.vgg16 import VGG16
import numpy as np

from imagenetData import ImageNetData


# Useful training functions
def validate_model(model):
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


def define_model(num_classes):
    # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    conv1 = layers.Conv2D(128, (1, 1), kernel_initializer='he_uniform')(model.layers[-1].output)
    conv2 = layers.Conv2D(64, (1, 1), kernel_initializer='he_uniform')(conv1)
    conv3 = layers.Conv2D(32, (1, 1), kernel_initializer='he_uniform')(conv2)
    flat1 = layers.Flatten()(conv3)
    class1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = layers.Dense(num_classes, activation='sigmoid')(class1)
    # define new model
    model = models.Model(inputs=model.inputs, outputs=output)

    # compile model
    model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    batch_size = 64
    dropout_rate = .2
    classes = [447, 96]  #  592, 950, 530,
    n_classes = len(classes)

    imageNet = ImageNetData(classes, images_per_class=500,
                            batch_size=batch_size,
                            validation_proportion=0.2,
                            augment_data=False)

    model_tf = define_model(n_classes)
    print(model_tf.summary())

    epochs = 30
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
            loss, acc = model_tf.test_on_batch(batch_data, y=batch_labels,
                                               sample_weight=None, reset_metrics=True)

            validation_accuracy, validation_loss = validate_model(model_tf)
            print(r'[Epoch %d, it %d] Training acc. %.3f, loss %.3f. \ Valid. acc. %.3f, loss %.3f' % (
                epoch,
                step,
                acc,
                loss,
                validation_accuracy,
                validation_loss
            ))
            val_acc_vals.append(validation_accuracy)
            test_accuracy = to_test_model(model_tf)
            test_acc_vals.append(test_accuracy)
            print("Time elapsed %.2f minutes" % ((time.time() - t_i) / 60.0))

        model_metrics = model_tf.train_on_batch(batch_data, y=batch_labels,
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
