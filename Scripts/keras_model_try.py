from __future__ import absolute_import, division, print_function, unicode_literals

import models_and_utils as mm
import tensorflow as tf
import matplotlib.pyplot as plt
from imagenetData import ImageNetData
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.callbacks import ModelCheckpoint

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


if __name__ == "__main__":
    batch_size = 64
    images_per_class = 500
    epochs = 30
    classes = [96, 950, 530]  # ,447, 530, 592, 950, 96]
    n_classes = len(classes)

    imageNet = ImageNetData(classes, images_per_class=500,
                            batch_size=batch_size, validation_proportion=0.4)

    model = mm.define_model(n_classes, use_pre_trained=True)

    checkpoint = ModelCheckpoint('best_model_val_loss.hdf5', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min', save_weights_only=True)

    imageNet.reset()

    x_train, y_train = imageNet.get_train_set()
    x_val, y_val = imageNet.get_validation_set()
    x_test, y_test = imageNet.get_test_set()

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                        callbacks=[checkpoint])

    print("Base accuracy in regular images : {}".format(model.evaluate(x=x_test, y=y_test, verbose=0)[1]))

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure()
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.figure()
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    plt.show()

    clear_session()
