"""
adversarial_generation.py: script that generates adversarial examples as described in the report. Resulting metrics are
stored in csv files
"""

import os

import pandas as pd
import Scripts.adversarial_utils as au
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from Scripts.utils import image_getter

tf.compat.v1.enable_eager_execution()

DICT_FILE_PATH = '../image_metadata/imagenet_class_index.json'
IMAGES_PATH = '../images'
NUM_CLASSES = 1000

if __name__ == "__main__":
    attacks_list = ['fgsm', 'rfgs', 'fg', 'step_ll']  # ataques a usar

    actual_num_classes = 915  # número de clases a usar
    images_per_class = 3  # númmero de imagenes por clase a usar

    res_model = ResNet50(weights='imagenet')
    res_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy(k=5)])

    img_folder_names = os.listdir(IMAGES_PATH)[:actual_num_classes]
    dirs = [os.path.join(IMAGES_PATH, img_dir) for img_dir in img_folder_names]

    img_folder_dirs = dict(zip(img_folder_names, dirs))

    labels = []
    images = []

    for img_folder_name in img_folder_names:
        n_class = int(img_folder_name.split("_")[0])
        labels += [n_class] * images_per_class
        images += image_getter(img_folder_dirs[img_folder_name])[:images_per_class]

    images = preprocess_input(np.array(images))
    one_hot_labels = tf.one_hot(labels, NUM_CLASSES)

    accuracy_list = []
    accuracy_5_list = []

    epsilons = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    plotted_idx = []

    for attack_type in attacks_list:
        for epsilon in tqdm(epsilons):
            if epsilon == 0:
                _, accu_1, accu_5 = res_model.evaluate(x=images.copy(), y=one_hot_labels)
            else:
                img_adversarial = next(au.simple_generate_adversarial(model=res_model, examples=images,
                                                                      labels=one_hot_labels, epsilon=epsilon,
                                                                      attack_type=attack_type))
                _, accu_1, accu_5 = res_model.evaluate(x=img_adversarial.copy(), y=one_hot_labels, batch_size=1)

            print(f'Accuracy top 1 (eps = {epsilon}) para ataque {attack_type}: {accu_1}')
            print(f'Accuracy top 5 (eps = {epsilon}) para ataque {attack_type}: {accu_5}')
            accuracy_list.append(accu_1)
            accuracy_5_list.append(accu_5)

        df = pd.DataFrame({'epsilons': epsilons, 'accuracy': accuracy_list, 'accuracy5': accuracy_5_list})
        df.to_csv(attack_type + '.csv')
