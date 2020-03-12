import numpy as np

from tensorflow.python.keras import layers, models, optimizers, losses
from tensorflow.python.keras.applications.vgg16 import VGG16


def define_model(num_classes, use_pre_trained=False):
    """
    It returns the model that'll be used for image classification.

    :param num_classes:         the number of classes that's beign classified
    :param use_pre_trained:     whether or not to use a pretrained model
    :return:                    a tensorflow.keras model
    """
    if use_pre_trained:
        return tl_model(num_classes)
    else:
        return own_model(num_classes)


def own_model(num_classes):
    """
    It defines the layers and structure of the CNN used to classify images.

    :param num_classes:     the number of classes that the model classify
    :return:                return a tensorflow.keras model
    """

    own_model_ = models.Sequential()

    own_model_.add(layers.Conv2D(16, (7, 7), activation='relu', padding='same', input_shape=(224, 224, 3)))
    own_model_.add(layers.MaxPooling2D((2, 2)))

    own_model_.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    own_model_.add(layers.MaxPooling2D((2, 2)))

    own_model_.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    own_model_.add(layers.MaxPooling2D((2, 2)))

    own_model_.add(layers.Conv2D(16, (7, 7), activation='relu', padding='same'))
    own_model_.add(layers.MaxPooling2D((2, 2)))

    own_model_.add(layers.Conv2D(num_classes, (1, 1)))
    own_model_.add(layers.GlobalAveragePooling2D())

    own_model_.summary()

    own_model_.compile(optimizer=optimizers.RMSprop(),
                       loss=losses.CategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])

    return own_model_


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
