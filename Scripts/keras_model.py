from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import time

from tensorflow.keras import datasets, layers, models
import numpy as np

from imagenetData import ImageNetData


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


def test(model):
    batches = imageNet.get_test_set(as_batches=True)
    accs = []
    for batch in batches:
        data, labels = batch
        _, acc = model.test_on_batch(data, y=labels,
                                          sample_weight=None, reset_metrics=True)
        accs.append(acc)
    mean_acc = np.array(accs).mean()
    return mean_acc


if __name__ == "__main__":
    # Load dataset
    batch_size = 64
    n_classes = 2
    dropout_rate = .2
    classes = [447, 530]  # , 592, 950, 96]
    imageNet = ImageNetData(classes, images_per_class=500,
                            batch_size=batch_size,
                            validation_proportion=0.2,
                            augment_data=False)

    # Defining the model
    model = models.Sequential()

    model.add(layers.Conv2D(32, (7, 7), activation='relu',
                            padding='same', input_shape=(224, 224, 3)))
    # model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(64, (5, 5), activation='relu',
                            padding='same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(64, (5, 5), activation='relu',
                            padding='same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(32, (9, 9), activation='relu',
                            padding='same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(n_classes, (1, 1)))
    model.add(layers.GlobalAveragePooling2D())

    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.SGD(nesterov=True),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

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
            test_accuracy = test(model)
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