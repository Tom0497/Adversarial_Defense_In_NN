from __future__ import absolute_import, division, print_function, unicode_literals

import models_and_utils as mm
import tensorflow as tf
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

    x_train, y_train = imageNet.get_train_set()
    x_val, y_val = imageNet.get_validation_set()
    x_test, y_test = imageNet.get_test_set()

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                        callbacks=[checkpoint])

    print("Base accuracy in regular images : {}".format(model.evaluate(x=x_test, y=y_test, verbose=0)[1]))

    mm.plot_learning_curves(history, epochs)

    clear_session()
