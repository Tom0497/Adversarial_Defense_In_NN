import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sys
import time

from DataExtractor import get_labels_for_wnid
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from utils import image_getter

tf.compat.v1.enable_eager_execution()


def adversarial_pattern(model, image, label):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)

    gradient = tape.gradient(loss, image)

    return gradient


def adversarial_step_ll(model, image, num_classes):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        y_ll = model(image).numpy().argmin()
        y_ll = labels_to_one_hot([y_ll], num_classes)[0]
        loss = tf.keras.losses.MSE(y_ll, prediction)

    gradient = tape.gradient(loss, image)

    signed_grad = -1 * tf.sign(gradient)

    return signed_grad


def labels_to_one_hot(original_labels, number_of_classes):
    """
    Converts list of integers to numpy 2D array with one-hot encoding

    :param number_of_classes:   how many classes are being classified
    :param original_labels:     the labels associated with some data in a vector form
    :return:                    one-hot encoding version of labels, therefore in a matrix
    """
    n = len(original_labels)
    one_hot_labels = np.zeros([n, number_of_classes], dtype=int)
    code_list = np.unique(original_labels)
    code_ix = np.argsort(code_list)
    code_dict = {code_list[i]: code_ix[i] for i in range(len(code_list))}
    labels_decoded = [code_dict[label] for label in original_labels]
    one_hot_labels[np.arange(n), np.asarray(labels_decoded)] = 1
    return one_hot_labels


if __name__ == "__main__":
    use_step_ll = False
    plot_resulting_examples = False
    attack_type = 'rfgs'

    model = ResNet50(weights='imagenet')
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy(k=5)])

    inputs = sys.argv
    current_directory = os.getcwd()
    directory_path = os.path.dirname(current_directory)
    CLASSES_FILE = "imagenet_class_index.json"
    DICT_FILE_PATH = os.path.join(directory_path, "image_metadata", CLASSES_FILE)
    images_path = os.path.dirname(current_directory) + "/images"

    folders_names = os.listdir(images_path)
    dirs = [images_path + "/" + img_dir + "/" for img_dir in folders_names]

    labels = []
    images = []

    actual_num_classes = 50  # cambiar aquí para cambiar el número de clases a usar
    images_per_class = 1           # cambiar aquí para cambiar el número de imágenes por clase a usar

    dirs = dirs[:actual_num_classes]

    for img_dir in dirs:
        n_class = int(img_dir.split("/")[-2].split("_")[0])
        labels += [n_class] * images_per_class

    num_classes = 1000
    one_hot_labels = labels_to_one_hot(labels, num_classes)

    accuracy_list = []
    accuracy_5_list = []

    epsilons = np.linspace(0, 10000, 6)

    for epsilon in epsilons:
        epsilon_og = epsilon
        alpha = epsilon / 2

        aciertos = []
        aciertos_5 = []

        img_idx = 0

        for img_dir in dirs:
            images = image_getter(img_dir)

            images = images[:images_per_class]

            assert (len(images) == images_per_class)

            for img in images:
                img = preprocess_input(img.copy())

                if epsilon == 0:
                    start = time.process_time()
                    y_pred = model.predict(img.copy()[np.newaxis, :])

                else:
                    start = time.process_time()
                    label = one_hot_labels[img_idx]

                    if use_step_ll:
                        perturbations = \
                            adversarial_step_ll(model, img.copy().reshape((1, 224, 224, 3)), num_classes).numpy()
                    else:
                        if attack_type == 'rfgs':
                            img += alpha * np.random.randn(img.shape[0], img.shape[1], img.shape[2])
                            epsilon -= alpha

                        output_gradient = adversarial_pattern(model, img.reshape((1, 224, 224, 3)), label)

                        if attack_type == 'fg':
                            perturbations = (output_gradient.numpy() / np.linalg.norm(output_gradient.numpy()))
                        else:
                            perturbations = tf.sign(output_gradient).numpy()

                    img_adversarial = img + perturbations * epsilon

                    img_adv = img_adversarial.copy()
                    img_adv = img_adv[0]
                    difference = img_adv.copy() - img.copy()

                    y_pred = model.predict(img_adv.copy()[np.newaxis, :])

                    if plot_resulting_examples:
                        plt.figure()

                        plt.subplot(1, 3, 1)
                        plt.imshow(img.copy() / 255)
                        plt.axis('off')

                        plt.subplot(1, 3, 2)
                        plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
                        plt.axis('off')

                        plt.subplot(1, 3, 3)
                        plt.imshow(img_adv / 255)
                        plt.axis('off')

                        plt.tight_layout()

                        plt.show()

                #pred, pred_5 = y_pred.argmax(), np.argpartition(np.squeeze(y_pred), -5)[-5:]
                pred_raw = decode_predictions(y_pred, top=1)[0][0]
                pred = int(get_labels_for_wnid(pred_raw[0], DICT_FILE_PATH)[0])
                pred_5_raw = decode_predictions(y_pred, top=5)[0]
                pred_5 = []
                for top5 in pred_5_raw:
                    pred_5.append(int(get_labels_for_wnid(top5[0], DICT_FILE_PATH)[0]))

                print(
                    f'Image {img_idx} out of {len(dirs) * images_per_class} (eps = {epsilon_og}). Elapsed time: {time.process_time() - start}')
                aciertos.append(labels[img_idx] == pred)
                aciertos_5.append(labels[img_idx] in pred_5)
                img_idx += 1

        accuracy = sum(aciertos) / len(aciertos)
        accuracy_5 = sum(aciertos_5) / len(aciertos_5)
        print(f'Accuracy top 1 (eps = {epsilon_og}): {accuracy}')
        print(f'Accuracy top 5 (eps = {epsilon_og}): {accuracy_5}')
        accuracy_list.append(accuracy)
        accuracy_5_list.append(accuracy_5)

    # import pandas as pd
    # df = pd.DataFrame({'epsilons': epsilons, 'accuracy': accuracy_list, 'accuracy5': accuracy_5_list})
    # df.to_csv(attack_type + '.csv')

    plt.plot(epsilons, accuracy_list, label='accuracy top-1')
    plt.plot(epsilons, accuracy_5_list, label='accuracy top-5')
    plt.show()
