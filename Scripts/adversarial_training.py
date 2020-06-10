"""
adversarial_training.py: script that applies adversarial training as described in the report. Variables that can be
                        adjusted are at the beginning of script, being the most relevant attack_type.
"""

import Scripts.adversarial_utils as au
import matplotlib.pyplot as plt
import Scripts.models_and_utils as mm
import numpy as np
import seaborn as sns
import tensorflow as tf
from Scripts.imagenet_data import ImageNetData
from tensorflow.python.keras.callbacks import ModelCheckpoint
from pickle import dump

tf.compat.v1.enable_eager_execution()

sns.reset_defaults()
sns.set(context='paper', style='ticks', font_scale=1.5)

if __name__ == "__main__":
    # define experiment parameters
    epochs = 50
    adv_lr = 0.0005
    batch_size = 64
    adv_val_prop = 0.5
    adv_test_num = 100
    adv_train_prop = 1
    attack_type = 'rfgs'
    train_data_prop = 0.6
    images_per_class = 500

    # classes to use in experiment
    classes = [96, 950, 530]  # Available classes : [447, 530, 592, 950, 96]
    n_classes = len(classes)

    # data manager object
    imageNet = ImageNetData(classes, images_per_class=images_per_class, validation_proportion=(1-train_data_prop))

    # load defined architecture and weights
    model = mm.define_model(n_classes, use_pre_trained=True, learning_rate=adv_lr)
    model.load_weights('../logs/weights/best_model_val_loss.hdf5')

    # get train, test and validation sets
    x_train, y_train = imageNet.get_train_set()
    x_test, y_test = imageNet.get_test_set()
    x_val, y_val = imageNet.get_validation_set()

    # number of adversarial images for training
    adv_train_num = int(len(x_train) * adv_train_prop)

    # get adversarial images for defense training
    x_adv_train, idxs_adv_train, y_adv_train = next(au.generate_adversarial(model=model,
                                                                            examples=x_train,
                                                                            labels=y_train,
                                                                            num_classes=n_classes,
                                                                            examples_num=adv_train_num,
                                                                            attack_type=attack_type))

    # selecting remaining images not adversarial
    x_train = np.delete(x_train, idxs_adv_train, axis=0)
    y_train = np.delete(y_train, idxs_adv_train, axis=0)

    # train set of mixed examples both adversarial and not
    x_train_set = np.r_[x_train, x_adv_train]
    y_train_set = np.r_[y_train, y_adv_train]

    # epsilon values to be used for testing the model
    epsilons = np.linspace(0, 3, num=13)

    x_adversarial_test_epsilons = []
    y_adversarial_test_epsilons = []
    adv_test_accu_no_defense = []
    adv_test_loss_no_defense = []

    # evaluation of the model without defense training
    for epsilon in epsilons:
        if epsilon != 0:
            # generation of test adversarial examples with epsilon
            x_adversarial_test, _, y_adversarial_test = next(au.generate_adversarial(model=model,
                                                                                     examples=x_test,
                                                                                     labels=y_test,
                                                                                     num_classes=n_classes,
                                                                                     epsilon=epsilon,
                                                                                     examples_num=adv_test_num,
                                                                                     attack_type=attack_type))

            # saving examples generated and measuring accuracy
            x_adversarial_test_epsilons.append(x_adversarial_test)
            y_adversarial_test_epsilons.append(y_adversarial_test)
            loss, accuracy = model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0)
        else:
            # accuracy in non-adversarial examples
            loss, accuracy = model.evaluate(x=x_test, y=y_test, verbose=0)

        # register of accuracy and loss and display
        adv_test_accu_no_defense.append(accuracy)
        adv_test_loss_no_defense.append(loss)
        print("Accuracy base, epsilon {0:.2f}: {1:.3f}".format(epsilon, accuracy))

    # number of adversarial images for validation
    adv_val_num = int(len(y_val) * adv_val_prop)

    # get adversarial images for defense training validation set
    x_adv_val, idxs_adv_val, y_adv_val = next(au.generate_adversarial(model=model,
                                                                      examples=x_val,
                                                                      labels=y_val,
                                                                      num_classes=n_classes,
                                                                      examples_num=adv_val_num,
                                                                      attack_type=attack_type))

    # selecting remaining images not adversarial
    x_val = np.delete(x_val, idxs_adv_val, axis=0)
    y_val = np.delete(y_val, idxs_adv_val, axis=0)

    # validation set of mixed examples both adversarial and not
    x_val_set = np.r_[x_val, x_adv_val]
    y_val_set = np.r_[y_val, y_adv_val]

    # checkpoint for saving best model weights based on minimum validation loss
    checkpoint = ModelCheckpoint(f'../logs/weights/defense/best_adv_model_{attack_type}.hdf5',
                                 monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min', save_weights_only=True)

    # fitting the model with adversarial training
    history = model.fit(x_train_set, y_train_set, batch_size=batch_size, epochs=epochs,
                        validation_data=(x_val_set, y_val_set), callbacks=[checkpoint])

    # using the best model from adversarial training
    model.load_weights(f'../logs/weights/defense/best_adv_model_{attack_type}.hdf5')

    # measuring accuracy on the model with adversarial training, same test sets
    adv_test_accu_after_epsilons = []
    adv_test_loss_after_epsilons = []
    for epsilon_index in range(len(epsilons)):
        if epsilon_index != 0:
            loss, accuracy = model.evaluate(x=x_adversarial_test_epsilons[epsilon_index - 1],
                                            y=y_adversarial_test_epsilons[epsilon_index - 1], verbose=0)
        else:
            loss, accuracy = model.evaluate(x=x_test, y=y_test, verbose=0)
        adv_test_accu_after_epsilons.append(accuracy)
        adv_test_loss_after_epsilons.append(loss)
        print("Accuracy fitted, epsilon {0:.2f}: {1:.3f}".format(epsilons[epsilon_index], accuracy))

    # comparison plot of accuracy before and after adversarial training
    plt.plot(epsilons, adv_test_accu_no_defense, 'o', label='Before adv. training')
    plt.plot(epsilons, adv_test_accu_after_epsilons, 'o', label='After adv. training')
    plt.legend()
    plt.show()

    # put relevant data into a dict
    info_dict = {'acc_no_def': adv_test_accu_no_defense,
                 'acc_def': adv_test_accu_after_epsilons,
                 'loss_no_def': adv_test_loss_no_defense,
                 'loss_def': adv_test_loss_after_epsilons,
                 'train_history': history.history}

    # save data for further use
    with open(f'../logs/history/defense/adv_def_hist_{attack_type}.pkl', 'wb') as f:
        dump(info_dict, f)
