import os
import time

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from Scripts.utils import image_getter, restore_original_image_from_array, plot_image_comparison
from Scripts.imagenetData import labels_to_one_hot

tf.compat.v1.enable_eager_execution()

DICT_FILE_PATH = '../image_metadata/imagenet_class_index.json'
IMAGES_PATH = '../images'
NUM_CLASSES = 1000

# TODO: in next two funcs, make it work by batches when len(images) > 20 approx, otherwise runs out of mem
# TODO: consider making only prediction by batches first, if fails then do whole process by batches, but internally


def adversarial_pattern(model, image, label):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)[0]
        loss = loss_object(label, prediction)

    gradient = tape.gradient(loss, image)

    return gradient


def adversarial_step_ll(model, image, num_classes):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        # y_ll = prediction.numpy().argmin(axis=1)
        # y_ll = tf.one_hot(y_ll, num_classes)
        # loss = loss_object(y_ll, prediction)
        y_ll = model(image).numpy().argmin()
        y_ll = labels_to_one_hot([y_ll], num_classes)[0]
        loss = tf.keras.losses.MSE(y_ll, prediction)

    signed_gradient = tape.gradient(loss, image)

    return -1 * tf.sign(signed_gradient)


def generate_adversarial(model, examples, labels, epsilon, attack_type):
    while True:
        x = []

        for idx, image in enumerate(tqdm(examples)):
            label = labels[idx]

            if attack_type == 'step_ll':
                perturbations = adversarial_step_ll(model, image.reshape((1, 224, 224, 3)), NUM_CLASSES).numpy()
                perturbations = np.array([pert / np.linalg.norm(pert) for pert in perturbations])

            elif attack_type == 'rfgs':
                alpha = epsilon/2
                image += alpha * np.random.randn(*image.shape)
                epsilon -= alpha
                perturbations = adversarial_pattern(model, image.reshape((1, 224, 224, 3)), label).numpy()
                perturbations = np.array([pert / np.linalg.norm(pert) for pert in perturbations])

            elif attack_type == 'fg':
                perturbations = adversarial_pattern(model, image.reshape((1, 224, 224, 3)), label).numpy()
                perturbations = np.array([pert / np.linalg.norm(pert) for pert in perturbations])

            elif attack_type == 'fgsm':
                perturbations = tf.sign(adversarial_pattern(model, image.reshape((1, 224, 224, 3)), label)).numpy()
                perturbations = np.array([pert / np.linalg.norm(pert) for pert in perturbations])

            else:
                print('No attack seems to fit with the attack given')
                break

            img_adversarial = image + 388 * perturbations * epsilon # 388 = (224*224*3)**0.5
            x.append(img_adversarial)

        x = np.asarray(x).reshape((len(examples), 224, 224, 3))

        yield x


if __name__ == "__main__":
    attacks_list = ['step_ll']
    for attack_type in attacks_list:
        plot_resulting_examples = False
        save_results = True

        actual_num_classes = 1000   # cambiar aqui para cambiar el numero de clases a usar
        images_per_class = 3      # cambiar aqui para cambiar el numero de imagenes por clase a usar

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

        #epsilons = np.linspace(0, 10, 11)
        epsilons = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
        plotted_idx = []

        for epsilon in tqdm(epsilons):
            alpha = epsilon / 2
            if epsilon == 0:
                _, accu_1, accu_5 = model.evaluate(x=images.copy(), y=one_hot_labels)
            else:
                img_adversarial = next(generate_adversarial(model=model, examples=images, labels=one_hot_labels,
                                                            epsilon=epsilon, attack_type=attack_type))
                _, accu_1, accu_5 = model.evaluate(x=img_adversarial.copy(), y=one_hot_labels, batch_size=1)

                epsilon = epsilon if attack_type != 'rfgs' else epsilon + alpha

                if plot_resulting_examples:
                    idx_ex = np.random.randint(0, len(images))
                    while idx_ex in plotted_idx:
                        idx_ex = np.random.randint(0, len(images))
                    plotted_idx.append(idx_ex)

                    img_example = images[idx_ex].copy()
                    img_adv_example = img_adversarial[idx_ex].copy()

                    _, reg_class, confidence = decode_predictions(model.predict(img_example[np.newaxis, :]))[0][0]
                    _, adv_class, adv_confidence = decode_predictions(model.predict(img_adv_example[np.newaxis, :]))[0][0]

                    img_example = restore_original_image_from_array(img_example)
                    img_adv_example = restore_original_image_from_array(img_adv_example)

                    plot_image_comparison(img_example, img_adv_example,
                                          title_img='Original: {0} \n Confidence= {1:.1f}%'.format(reg_class,
                                                                                                   confidence*100),
                                          title_adv='Adversarial: {0} \n Confidence= {1:.1f}%'.format(adv_class,
                                                                                                      adv_confidence*100),
                                          title_diff='Difference \n epsilon= {}'.format(epsilon), save_plot=[False,''])

            print(f'Accuracy top 1 (eps = {epsilon}): {accu_1}')
            print(f'Accuracy top 5 (eps = {epsilon}): {accu_5}')
            accuracy_list.append(accu_1)
            accuracy_5_list.append(accu_5)

        import pandas as pd
        df = pd.DataFrame({'epsilons': epsilons, 'accuracy': accuracy_list, 'accuracy5': accuracy_5_list})
        df.to_csv(attack_type + '6.csv')

        plt.plot(epsilons, accuracy_list, label='accuracy top-1')
        plt.plot(epsilons, accuracy_5_list, label='accuracy top-5')
        plt.show()
