import numpy as np
import os
import tensorflow as tf
import time
import glob
import sys

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as image_keras
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.callbacks import TensorBoard
from Scripts.attackImplemented import fastGradientAttack
from multiprocessing import Pool

current_directory = os.getcwd()
images_path = os.path.dirname(current_directory) + r"/images/"
WIDTH, HEIGHT = 224, 224


def correct_predictions(model, image, epsilon, attack_type='fgsm'):
    img_adv = fastGradientAttack(model, image, epsilon, epsilon * .5,
                                 attack_type=attack_type, preprocess=True)

    adv_label = model.predict(preprocess_input(img_adv.copy() * 255)[np.newaxis, :])

    return adv_label.argmax(), np.argpartition(np.squeeze(adv_label), -5)[-5:]


def image_getter(path):
    """
    Given a path to a image folder along with an specified extension, it reads al the images that fits the extension
    an puts them into a list

    :param path:        the path to folder along with the image extension
    :return:            a list of images
    """
    image_list = []
    for filename in glob.glob(path):
        try:
            im = image_keras.load_img(filename, target_size=(WIDTH, HEIGHT))
            return [image_keras.img_to_array(im)]
        except IOError as e:
            print(e)
    return image_list


if __name__ == "__main__":
    """
    tensorboard = TensorBoard(log_dir=os.getcwd()+"/Scripts/log",
                              histogram_freq=0,
                              write_graph=True,
                              write_images=False)
    """

    model = ResNet50(weights='imagenet')
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    """
    plot_model(model,
               to_file='model1.png',
               show_layer_names=False,
               show_shapes=False)
    """
    inputs = sys.argv
    max_classes = int(inputs[1])

    folders_names = os.listdir(images_path)[:max_classes]
    dirs = [images_path + img_dir + r"/*.jpg" for img_dir in folders_names]
    folder_label = [int(img_dir.split("_")[0]) for img_dir in folders_names]

    with Pool() as p:
        images = p.map(image_getter, dirs)
        num_labels = [len(img_folder) for img_folder in images]
        images = [img for sublist in images for img in sublist]

    y = []
    for i, value in enumerate(num_labels):
        label = folder_label[i]
        y += [label for _ in range(value)]

    epsilon = float(inputs[2])
    attack_type = inputs[3]
    if attack_type not in ['fg', 'fgsm', 'rfgs']:
        attack_type = 'fgsm'

    accuracy_list = []
    accuracy_5_list = []
    aciertos = []
    aciertos_5 = []

    for index, img in enumerate(images):
        ywut = y[index]
        start = time.process_time()
        pred, pred_5 = correct_predictions(model, img.copy(), epsilon, attack_type=attack_type)
        print(f'Image {index} out of {len(images)} (eps = {epsilon}). Elapsed time: {time.process_time() - start}')
        aciertos.append(ywut == pred)
        aciertos_5.append(ywut in pred_5)
    accuracy = sum(aciertos) / len(aciertos)
    accuracy_5 = sum(aciertos_5) / len(aciertos_5)
    print(f'Accuracy top 1 (eps = {epsilon}): {accuracy}')
    print(f'Accuracy top 5 (eps = {epsilon}): {accuracy_5}')
    accuracy_list.append(accuracy)
    accuracy_5_list.append(accuracy_5)
