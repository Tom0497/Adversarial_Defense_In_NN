import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from absl import logging
from Scripts.foolbox_image_generator import image_getter
from tensorflow.keras.utils import plot_model


logging._warn_preinit_stderr = 0

NUM_PARALLEL_EXEC_UNITS = 4
os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_AFFINITY"] = "none"

current_directory = os.getcwd()
images_path = os.path.dirname(current_directory) + r"/images/"


if __name__ == "__main__":
    model = ResNet50(weights='imagenet')
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    plot_model(model,
               to_file='model.png',
               show_layer_names=True,
               show_shapes=True)

    loss = []
    accuracy = []

    for subdir, dirs, files in os.walk(images_path):
        for img_dir in dirs:

            folder_label = int(img_dir.split("_")[0])

            images, image_names = image_getter(images_path + img_dir + r"/*.jpg")
            images_preprocessed = np.asarray([preprocess_input(img.copy()) for img in images])

            if 0 in images_preprocessed.shape:
                continue

            y_real = np.ones(len(images), dtype=int)*folder_label

            loss_, accuracy_ = model.test_on_batch(images_preprocessed, y_real)

            loss.append(loss_)
            accuracy.append(accuracy_)

    loss = np.asarray(loss)
    accuracy = np.asarray(accuracy)

    print('Loss en dataset : %.3f +/- %.3f' % (loss.mean(), loss.std()))
    print('Accuracy en dataset : %.3f +/- %.3f' % (accuracy.mean(), accuracy.std()))


