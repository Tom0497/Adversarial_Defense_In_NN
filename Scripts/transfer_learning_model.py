from __future__ import absolute_import, division, print_function, unicode_literals

import time

import models_and_utils as mm
import numpy as np
import tensorflow as tf
from imagenetData import ImageNetData
from tensorflow.python.keras.backend import clear_session

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

if __name__ == "__main__":
    batch_size = 64
    dropout_rate = .2
    classes = [447, 96]  #  592, 950, 530,
    n_classes = len(classes)

    imageNet = ImageNetData(classes, images_per_class=500,
                            batch_size=batch_size,
                            validation_proportion=0.2,
                            augment_data=False)

    model_tf = mm.define_model(n_classes)

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

            validation_accuracy, validation_loss = mm.validate_model(model_tf, imageNet)
            print(r'[Epoch %d, it %d] Training acc. %.3f, loss %.3f. \ Valid. acc. %.3f, loss %.3f' % (
                epoch,
                step,
                acc,
                loss,
                validation_accuracy,
                validation_loss
            ))
            val_acc_vals.append(validation_accuracy)
            test_accuracy = mm.to_test_model(model_tf)
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

    clear_session()
