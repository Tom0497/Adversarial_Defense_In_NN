"""
visual_attack_comparison.py: script that generates report's Figure 5
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from Scripts.utils import image_getter, restore_original_image_from_array, plot_image_comparison
from Scripts.adversarial_utils import simple_generate_adversarial

tf.compat.v1.enable_eager_execution()

DICT_FILE_PATH = '../image_metadata/imagenet_class_index.json'
IMAGES_PATH = '../images'
NUM_CLASSES = 1000

actual_num_classes = 1000
images_per_class = 3

model = ResNet50(weights='imagenet')
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
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

attacks_lists = ['fgsm', 'fg', 'rfgs', 'step_ll']

idx_ex = 94

epsilon = 3

images_red = [images[idx_ex]]
labels_red = [one_hot_labels[idx_ex]]

for attack_type in attacks_lists:
    img_example = images[idx_ex].copy()

    img_adversarial = next(simple_generate_adversarial(model=model, examples=images_red, labels=labels_red, epsilon=epsilon, attack_type=attack_type))

    img_adv_example = img_adversarial[0].copy()

    _, reg_class, confidence = decode_predictions(model.predict(img_example[np.newaxis, :]))[0][0]
    _, adv_class, adv_confidence = decode_predictions(model.predict(img_adv_example[np.newaxis, :]))[0][0]

    img_example = restore_original_image_from_array(img_example)
    img_adv_example = restore_original_image_from_array(img_adv_example)

    filename = f'plot_index_{idx_ex}_epsilon_{epsilon}_{attack_type}'

    plot_image_comparison(img_example, img_adv_example,
                          title_img='Original: {0} \n Confidence= {1:.1f}%'.format(reg_class,
                                                                                   confidence*100),
                          title_adv='Adversarial: {0} \n Confidence= {1:.1f}%'.format(adv_class,
                                                                                      adv_confidence*100),
                          title_diff='Difference \n $\epsilon$= {}'.format(epsilon), save_plot=[True, filename])

