"""
model_training.py: script for GridSearch of best hyper-parameters of model training and training of models.
                 It register the weights of best models and also metadata from training.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from pickle import dump

import models_and_utils as mm
import tensorflow as tf
from imagenetData import ImageNetData
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.callbacks import ModelCheckpoint

if __name__ == "__main__":
    # Training models parameters, for GridSearch small number of epochs is recommended
    epochs = 15
    batch_size = 64
    images_per_class = 500
    classes = [96, 950, 530]  # [447, 530, 592, 950, 96]
    n_classes = len(classes)

    # Learning rates and optimizers to try
    learning_rates = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]  # [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    optimizers_names = ['sgd', 'adam', 'rmsprop', 'adagrad']  # ['sgd', 'adam', 'rmsprop', 'adagrad']

    # data manager object
    imageNet = ImageNetData(classes, images_per_class=500, validation_proportion=0.4)

    # get train, test and validation sets
    x_train, y_train = imageNet.get_train_set()
    x_val, y_val = imageNet.get_validation_set()
    x_test, y_test = imageNet.get_test_set()

    # training loop over all combinations of hyper-parameters
    register = {}
    for opt in optimizers_names:
        for lr in learning_rates:
            # gpu configurations for several runs
            config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
            config.gpu_options.allow_growth = True
            session = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(session)

            print('\n'*2, '*'*30)
            print('Results for {0} optimizer and learning rate {1}'.format(opt, lr))
            print('*'*30, '\n'*2)

            # define the model with hyper-parameters (opt, lr)
            model = mm.define_model(n_classes, optimizer=opt, learning_rate=lr, use_pre_trained=True)
            model_str = 'model_{0}_{1}'.format(opt, str(lr).replace('.', ''))

            # checkpoint callback for saving best model in terms of validation loss
            checkpoint = ModelCheckpoint('../logs/weights/base_model/{0}epochs/best_{1}.hdf5'.format(epochs, model_str),
                                         verbose=1, monitor='val_loss', save_best_only=True, mode='min',
                                         save_weights_only=True)

            # time callback for measuring time of execution of model training
            time_callback = mm.TimeHistory()

            # fitting the model to the data
            history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                                validation_data=(x_val, y_val), callbacks=[checkpoint, time_callback])

            # evaluation in test set
            test_results = model.evaluate(x=x_test, y=y_test, verbose=0)

            # clearing session for training of next model in loop
            for _ in range(10):
                clear_session()

            # few additions to history of model training for futher use
            history.history['time_history'] = time_callback.get_logs()
            history.history['training_time'] = time_callback.get_training_time()
            history.history['test_results'] = test_results

            # plot learning curves and saving history into register
            # mm.plot_learning_curves(history.history)
            register[model_str] = history.history

    # save register of trained models
    with open(f'../logs/history/base_model/{epochs}epochs/models_hist.pkl', 'wb') as f:
        dump(register, f)
