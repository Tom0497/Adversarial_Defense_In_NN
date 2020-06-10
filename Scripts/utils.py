import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.preprocessing import image

HEIGHT, WIDTH = 224, 224


def image_getter(path):
    """
    Given a path to a image folder along with an specified extension, it reads al the images that fits the extension
    an puts them into a list

    :param path:        the path to folder along with the image extension
    :return:            a list of images
    """
    image_list = []
    image_name = []
    for filename in glob.glob(os.path.join(path, '*')):
        try:
            im = image.load_img(filename, target_size=(WIDTH, HEIGHT))
            image_list.append(image.img_to_array(im))
            image_name.append(filename)
        except IOError as e:
            print(e)
    return image_list


def plot_image_comparison(image_, adversarial, save_plot, title_img="", title_adv="", title_diff="", ):
    """
    Plots an image_, its given adversarial generated example and the difference between them in order to visualize
    the similarities

    :param title_diff:
    :param title_adv:
    :param title_img:
    :param image_:           the image
    :param adversarial:     the adversarial example of the image_, previously generated
    """
    if image_ is not None:
        plt.figure()

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.subplot(1, 3, 1)
        plt.title('Original' if title_img == "" else title_img)
        plt.imshow(image_ / 255)  # division by 255 to convert [0, 255] to [0, 1]
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Adversarial' if title_adv == "" else title_adv)
        plt.imshow(adversarial / 255)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Difference' if title_diff == "" else title_diff)
        difference = adversarial - image_
        plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
        plt.axis('off')

        plt.tight_layout()
        if save_plot[0]:
            plt.savefig(save_plot[1] + '.eps', format='eps', bbox_inches='tight',pad_inches=0)
        plt.show()


def restore_original_image_from_array(x, data_format=None):
    """
    It reverses the pre processing of the images that was needed before passing the images to the network

    :param x:               the image that will be processed
    :param data_format:     indicates if channels are the first dimension of the image array
    :return:                the processed image in its original format
    """
    mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return np.clip(x, 0, 255)
