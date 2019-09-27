import cv2 as cv
import matplotlib.pyplot as plt
import os

current_directory = os.getcwd()
images_path = os.path.dirname(current_directory) + r"\images/"
specific = True
if specific:
    images_path += "n00005787/"


def ims_plot(path, limit=5):
    """
    Plots a set of images contained in the given path, until a limit is reached

    :param path: path were the images are
    :param limit: maximum number of images to be plotted
    :return:
    """
    for i, filename in enumerate(os.listdir(path)):
        filename = os.path.join(path, filename)
        if i == limit:
            break
        plt.figure()
        print(filename)
        image = cv.imread(filename, 1)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    ims_plot(images_path)
