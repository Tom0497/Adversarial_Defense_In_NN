import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
from tensorflow.python.keras import layers, models, optimizers, losses, callbacks
from tensorflow.python.keras.applications.vgg16 import VGG16


def define_model(num_classes, use_pre_trained=False, optimizer='sgd', learning_rate=0.001, dropout_rate=0.2,
                 display_summary=False):
    """
    It returns the model that'll be used for image classification. In any of both possible cases, the optimizer is
    SGD with Nesterov momentum.

    :param display_summary:     bool to indicate whether or not to display the architecture of the CNN used
    :param optimizer:           the optimization method to train the model, options are:
                                                                                * adam    --> ADAM
                                                                                * sgd     --> SGD
                                                                                * rmsprop --> RMSprop
                                                                                * adagrad --> ADAGrad
    :param learning_rate:       the learning rate used for the optimizer
    :param dropout_rate:        the dropout rate in case it applies
    :param num_classes:         the number of classes that's beign classified
    :param use_pre_trained:     whether or not to use a pretrained model
    :return:                    a tensorflow.keras model
    """
    if use_pre_trained:
        return tl_model(num_classes, optimizer=optimizer, learning_rate=learning_rate, summary=display_summary)
    else:
        return own_model(num_classes, optimizer=optimizer, dropout_rate=dropout_rate, learning_rate=learning_rate,
                         summary=display_summary)


def own_model(num_classes, optimizer, dropout_rate, learning_rate, summary):
    """
    It defines the structure and layers of the CNN model used for classification, it has a MLP as
    classifier at its bottom.

    :param summary:         bool to indicate whether or not to display the architecture of the CNN used
    :param optimizer:       the optimization method to train the model
    :param learning_rate:   the learning_rate used for the optimizer.
    :param num_classes:     the number of classes that the model classify
    :param dropout_rate:    sets the dropout rate for the dense layers of the model
    :return:                return a tensorflow.keras model
    """
    model = models.Sequential()

    model.add(layers.Conv2D(16, (7, 7), activation='relu', padding='same', input_shape=(224, 224, 3),
                            kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (7, 7), activation='relu', padding='same',
                            kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',
                            kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (7, 7), activation='relu', padding='same',
                            kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (1, 1), activation='relu', kernel_initializer='he_uniform'))

    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))

    model.add(layers.Dense(num_classes, activation='sigmoid', kernel_initializer='he_uniform'))

    if summary:
        model.summary()

    model.compile(optimizer=get_optimizer(optimizer, learning_rate),
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def tl_model(num_classes, optimizer, learning_rate, summary):
    """
    It gives a model based on a pre-trained classification model like VGG16 for example, giving
    the possibility of using a big model without training it.

    :param summary:         bool to indicate whether or not to display the architecture of the CNN used
    :param optimizer:       the optimization method to train the model
    :param learning_rate:   the learning rate used for the SGD optimizer.
    :param num_classes:     the number of classes that the model classify
    :return:                return a tensorflow.keras model
    """

    model = VGG16(include_top=False, input_shape=(224, 224, 3))

    for layer in model.layers:
        layer.trainable = False

    conv1 = layers.Conv2D(128, (1, 1), kernel_initializer='he_uniform')(model.layers[-1].output)
    conv2 = layers.Conv2D(64, (1, 1), kernel_initializer='he_uniform')(conv1)
    conv3 = layers.Conv2D(32, (1, 1), kernel_initializer='he_uniform')(conv2)

    flat1 = layers.Flatten()(conv3)
    flat2 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = layers.Dense(num_classes, activation='sigmoid')(flat2)

    model = models.Model(inputs=model.inputs, outputs=output)

    if summary:
        model.summary()

    model.compile(optimizer=get_optimizer(optimizer, learning_rate),
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def get_optimizer(optimizer, learning_rate):
    """
    A sort of swith-case for choosing the optimizer for a model.

    :param optimizer:       the string key for an optimizer
    :param learning_rate:   the learning rate that'll be used for training
    :return:                a keras.optimizers.optimizer object
    """
    optimizer_options = {'sgd': optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True),
                         'adam': optimizers.Adam(lr=learning_rate),
                         'rmsprop': optimizers.RMSprop(lr=learning_rate),
                         'adagrad': optimizers.Adagrad(lr=learning_rate)}

    try:
        return optimizer_options[optimizer]
    except KeyError:
        return optimizer_options['sgd']


def validate_model(classification_model, data_pipeline):
    """
    A function to validate a classification model through selecting a part of the dataset not used for training,
    de data is provided by the pipeline.

    :param classification_model:    the model that's being evaluated
    :param data_pipeline:           an object that handles the data for the model
    :return:                        mean accuracy and mean loss in the validation set
    """
    data_pipeline.shuffle_validation()
    batches = data_pipeline.get_validation_set(as_batches=True)
    accs = []
    xent_vals = []
    for batch in batches:
        data, labels = batch
        xentropy_val, acc = classification_model.test_on_batch(data, y=labels,
                                                               sample_weight=None, reset_metrics=True)
        accs.append(acc)
        xent_vals.append(xentropy_val)
    mean_xent = np.array(xent_vals).mean()
    mean_acc = np.array(accs).mean()
    return mean_acc, mean_xent


def to_test_model(classification_model, data_pipeline):
    """
    A function to test the model in certain stages of training, in order to see its improvement.

    :param classification_model:    the model that's being evaluated
    :param data_pipeline:           an object that handles the data for the model
    :return:                        mean accuracy in the test set
    """
    batches = data_pipeline.get_test_set(as_batches=True)
    accs = []
    for batch in batches:
        data, labels = batch
        _, acc = classification_model.test_on_batch(data, y=labels,
                                                    sample_weight=None, reset_metrics=True)
        accs.append(acc)
    mean_acc = np.array(accs).mean()
    return mean_acc


def plot_learning_curves(model_history, show_train_time=False, opt=None, lr=None):
    """
    It plots the learning curves of a model through its model history and the number of epochs used
    to train it, in specific utilizes both the train and validation loss and accuracy to show the
    progress of a model in its training stage.

    :param lr:                  the learning rate used for training, for title purpose
    :param opt:                 the optimizer's name used for training, for title purpose
    :param show_train_time:     bool, indicates whether or not to plot training time vs epoch.
    :param model_history:       a dict which contains the history of the model, loss and accuracy.
    """
    acc = model_history['acc']
    val_acc = model_history['val_acc']
    test_acc = model_history['test_results'][1]

    loss = model_history['loss']
    val_loss = model_history['val_loss']
    test_loss = model_history['test_results'][0]

    epoch_time = model_history['time_history']
    train_time = model_history['training_time']

    epochs = len(acc)
    epochs_range = range(epochs)

    title_append = '. {0}, lr={1}'.format(opt, lr)
    if opt is None or lr is None:
        title_append = ''

    sns.reset_defaults()
    sns.set(context='paper', style='ticks', font_scale=1.5, rc={'axes.grid': True})

    plt.figure()
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy' + title_append)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(epochs_range, loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss' + title_append)
    plt.tight_layout()
    plt.show()

    if show_train_time:
        plt.figure()
        plt.plot(epochs_range, epoch_time, marker='o')
        plt.hlines(train_time/epochs, epochs_range[0], epochs_range[-1], label='average', colors='r', alpha=.5)
        plt.xlabel('Epoch')
        plt.ylabel('Time (Sec)')
        plt.legend()
        plt.title('Training time per epoch' + title_append)
        plt.tight_layout()
        plt.show()

    print('Loss and accuracy in test set : loss {0:.4f} accuracy {1:.4f}'.format(test_loss, test_acc))
    print('Total time for training {0} epochs : {1:5.2f} min'.format(epochs, train_time/60))


def get_opt_and_eps(model_name_str):
    """
    A simple function to get optimizer's name an learning rate from a file's name.

    :param model_name_str:      the model's name, which has implicitly the info needed
    :return:                    optimizer's name and learning rate
    """
    parts = model_name_str.split('_')
    opt_dict = {'sgd': 'SGD', 'adam': 'Adam', 'rmsprop': 'RMSprop', 'adagrad': 'AdaGrad'}
    opt = opt_dict[parts[1]]

    str_eps = parts[2]
    str_eps = str_eps[0] + '.' + str_eps[1:]
    eps = float(str_eps)

    return opt, eps


def summary_of_register(register):
    """
    It displays the info of training from lots of models, when the info is in a dict.

    :param register:    a dict that contains all histories of different model's training
    """
    for model_name, history in register.items():
        print('\n' * 2, '*' * 30)
        print('Results for the model {}'.format(model_name))
        print('*' * 30)

        opt_str, lr = get_opt_and_eps(model_name)

        plot_learning_curves(history, opt=opt_str, lr=lr)


class TimeHistory(callbacks.Callback):
    """
    A small callback class in order to have a registry of the time it took the model to train,
    both by epoch and in total.
    """
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

    def get_logs(self):
        return self.times

    def get_training_time(self):
        return sum(self.times)
