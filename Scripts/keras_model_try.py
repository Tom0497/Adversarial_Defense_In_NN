from __future__ import absolute_import, division, print_function, unicode_literals

import models_and_utils as mm
import tensorflow as tf
from pickle import dump
from imagenetData import ImageNetData
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.callbacks import ModelCheckpoint

if __name__ == "__main__":
    batch_size = 64
    images_per_class = 500
    epochs = 15
    classes = [96, 950, 530]  # [447, 530, 592, 950, 96]
    n_classes = len(classes)

    learning_rates = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    optimizers_names = ['sgd', 'adam', 'rmsprop', 'adagrad']

    imageNet = ImageNetData(classes, images_per_class=500,
                            batch_size=batch_size, validation_proportion=0.4)

    x_train, y_train = imageNet.get_train_set()
    x_val, y_val = imageNet.get_validation_set()
    x_test, y_test = imageNet.get_test_set()

    register = {}
    for opt in optimizers_names:
        for lr in learning_rates:
            config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
            config.gpu_options.allow_growth = True
            session = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(session)

            print('\n'*2, '*'*30)
            print('Results for {0} optimizer and learning rate {1}'.format(opt, lr))
            print('*'*30, '\n'*2)

            model = mm.define_model(n_classes, optimizer=opt, learning_rate=lr, use_pre_trained=True)
            model_name = 'model_{0}_{1}'.format(opt, str(lr).replace('.', ''))

            checkpoint = ModelCheckpoint('trainModelsWeights/best_{}.hdf5'.format(model_name), verbose=1,
                                         monitor='val_loss', save_best_only=True, mode='min', save_weights_only=True)
            time_callback = mm.TimeHistory()

            history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                                validation_data=(x_val, y_val), callbacks=[checkpoint, time_callback])

            test_results = model.evaluate(x=x_test, y=y_test, verbose=0)

            for _ in range(10):
                clear_session()

            history.history['time_history'] = time_callback.get_logs()
            history.history['training_time'] = time_callback.get_training_time()
            history.history['test_results'] = test_results

            # print('Results for {0} optimizer and learning rate {1}'.format(opt, lr))

            # mm.plot_learning_curves(history.history)
            register[model_name] = history.history

    f = open('trainHistoryLogs/train_models_hist.pkl', 'wb')
    dump(register, f)
    f.close()
