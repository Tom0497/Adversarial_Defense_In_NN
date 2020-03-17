import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import layers, models, optimizers, losses
from tensorflow.python.keras.applications.vgg16 import VGG16


def define_model(num_classes, own_num=0, use_pre_trained=False, dropout_rate=0.2):
    """
    It returns the model that'll be used for image classification.

    :param dropout_rate:        the dropout rate in case it applies
    :param own_num:             it specifies which own model to use
    :param num_classes:         the number of classes that's beign classified
    :param use_pre_trained:     whether or not to use a pretrained model
    :return:                    a tensorflow.keras model
    """
    if use_pre_trained:
        return tl_model(num_classes)
    else:
        if own_num == 0:
            return own_model(num_classes)
        elif own_num == 1:
            return own_model_1(num_classes, dropout_rate=dropout_rate)


def own_model(num_classes):
    """
    It defines the layers and structure of the CNN used to classify images.

    :param num_classes:     the number of classes that the model classify
    :return:                return a tensorflow.keras model
    """

    own_model_ = models.Sequential()

    own_model_.add(layers.Conv2D(16, (7, 7), activation='relu', padding='same',
                                 input_shape=(224, 224, 3), kernel_initializer='he_uniform'))
    own_model_.add(layers.MaxPooling2D((2, 2)))

    own_model_.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='he_uniform'))
    own_model_.add(layers.MaxPooling2D((2, 2)))

    own_model_.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))
    own_model_.add(layers.MaxPooling2D((2, 2)))

    own_model_.add(layers.Conv2D(16, (7, 7), activation='relu', padding='same', kernel_initializer='he_uniform'))
    own_model_.add(layers.MaxPooling2D((2, 2)))

    own_model_.add(layers.Conv2D(num_classes, (1, 1)))
    own_model_.add(layers.GlobalAveragePooling2D())

    own_model_.summary()

    own_model_.compile(optimizer=optimizers.RMSprop(),
                       loss=losses.CategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])

    return own_model_


def own_model_1(num_classes, dropout_rate):
    """
    It defines the structure and layers of the CNN model used for classification, it has a MLP as
    classifier at its bottom.

    :param num_classes:     the number of classes that the model classify
    :param dropout_rate:    sets the dropout rate for the dense layers of the model
    :return:                return a tensorflow.keras model
    """
    own_model_1_ = models.Sequential()

    own_model_1_.add(layers.Conv2D(16, (7, 7), activation='relu', padding='same', input_shape=(224, 224, 3),
                                   kernel_initializer='he_uniform'))
    own_model_1_.add(layers.MaxPooling2D((2, 2)))

    own_model_1_.add(layers.Conv2D(32, (7, 7), activation='relu', padding='same',
                                   kernel_initializer='he_uniform'))
    own_model_1_.add(layers.MaxPooling2D((2, 2)))

    own_model_1_.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',
                                   kernel_initializer='he_uniform'))
    own_model_1_.add(layers.MaxPooling2D((2, 2)))

    own_model_1_.add(layers.Conv2D(128, (7, 7), activation='relu', padding='same',
                                   kernel_initializer='he_uniform'))
    own_model_1_.add(layers.MaxPooling2D((2, 2)))

    own_model_1_.add(layers.Conv2D(64, (1, 1), activation='relu', kernel_initializer='he_uniform'))

    own_model_1_.add(layers.Flatten())
    own_model_1_.add(layers.Dropout(dropout_rate))
    own_model_1_.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))

    own_model_1_.add(layers.Dense(num_classes, activation='sigmoid', kernel_initializer='he_uniform'))

    own_model_1_.summary()

    own_model_1_.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True),
                         loss="categorical_crossentropy",
                         metrics=['accuracy'])

    return own_model_1_


def tl_model(num_classes):
    """
    It gives a model based on a pre-trained classification model like VGG16 for example, giving
    the possibility of using a big model without training it.

    :param num_classes:     the number of classes that the model classify
    :return:                return a tensorflow.keras model
    """

    tl_model_ = VGG16(include_top=False, input_shape=(224, 224, 3))

    for layer in tl_model_.layers:
        layer.trainable = False

    conv1 = layers.Conv2D(128, (1, 1), kernel_initializer='he_uniform')(tl_model_.layers[-1].output)
    conv2 = layers.Conv2D(64, (1, 1), kernel_initializer='he_uniform')(conv1)
    conv3 = layers.Conv2D(32, (1, 1), kernel_initializer='he_uniform')(conv2)

    flat1 = layers.Flatten()(conv3)
    flat2 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = layers.Dense(num_classes, activation='sigmoid')(flat2)

    tl_model_ = models.Model(inputs=tl_model_.inputs, outputs=output)

    tl_model_.summary()

    tl_model_.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True),
                      loss=losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    return tl_model_


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


def plot_learning_curves(model_history, model_epochs):
    """
    It plots the learning curves of a model through its model history and the number of epochs used
    to train it, in specific utilizes both the train and validation loss and accuracy to show the
    progress of a model in its training stage.

    :param model_history:       a dict which contains the history of the model, loss and accuracy.
    :param model_epochs:        an int which tells for how many epochs the model was trained.
    """
    acc = model_history.history['acc']
    val_acc = model_history.history['val_acc']

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs_range = range(model_epochs)

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
