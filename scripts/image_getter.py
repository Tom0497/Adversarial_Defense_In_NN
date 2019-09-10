import pandas as pd
import os
from PIL import Image
import urllib.request
import urllib.error
import http.client

classes_file = r"\imagenet_class_index.json"
urls_file = r"\fall11_urls.txt"
current_directory = os.getcwd()
dict_path_ = os.path.dirname(current_directory) + r"\image_metadata" + classes_file
urls_path = os.path.dirname(current_directory) + r"\image_metadata" + urls_file
images_path = os.path.dirname(current_directory) + r"\images"


def get_dict_classes(path):
    """
    Creates a table with the 1000 classes names and their respective wnid code from the ImageNet dataset

    :param path: path where the .json file that contains the info is
    :return: pandas DataFrame where the columns contains both the wnid code and the class associated with it
    """
    with open(path) as json_file:
        data = pd.read_json(json_file).T
        columns_names = ("wnid", "class")
        data.columns = columns_names
    return data


def get_urls_and_wnid(path, limit=1024):
    """
    Formats the file located in path in order to get a table with wnid code and url of the ImageNet's DataSet.

    :param limit: maximum number of bytes to be read from file
    :param path: path to the .txt file that contains the raw version of the wnid and url data for the images
    :return: pandas DataFrame where the columns contains both de wnid code and the associated url of images
    """
    with open(path, encoding="latin-1") as file:
        data = file.readlines(limit)
        data = [line.strip().split("\t") for line in data]
        data = pd.DataFrame(data, columns=("wnid", "url"))
        data["wnid"] = data["wnid"].apply((lambda x: x.partition("_")[0]))
    return data


def get_image_from_url(url):
    """
    Makes a request to get an image from the given url direction

    :param url: the url direction where an image is located
    :return: an image or None in case of error
    """
    try:
        image = Image.open(urllib.request.urlopen(url))
        return image
    except (urllib.error.HTTPError,
            urllib.error.URLError,
            IOError,
            http.client.HTTPException) as e:
        print(e)
        return None


def save_image_in_path(image, image_name, path=os.getcwd()):
    """
    Saves a given picture into a folder specified in path, with the name passed

    :param image: the source image to be saved
    :param image_name: the name that the image'll be assigned
    :param path: the folder path where to save the image
    :return:
    """
    image.save(os.path.join(path, image_name), 'JPEG')


def get_images_from_urls(urls):
    """
    Given a list of urls directions, get all the images within them

    :param urls: the urls directions where the images are
    :return: a python list with all images that were able to get
    """
    images = []
    for url in urls:
        image = get_image_from_url(url)
        if image is not None:
            images.append(image)
        else:
            pass
    return images


def save_images_in_path(images, names, path=os.getcwd()):
    """
    Saves a set of images into a given folder path with the names passed

    :param images: the images to be saved
    :param names: the names associated with each image
    :param path: the folder path where to save the images
    :return:
    """
    assert len(images) == len(names)
    for image, name in zip(images, names):
        image.save(os.path.join(path, name), 'JPEG')


if __name__ == "__main__":
    data_ = get_urls_and_wnid(urls_path)
    urls_ = data_["url"].values
    images_ = get_images_from_urls(urls_)
    names_ = ["image{}".format(i) + ".jpg" for i in range(len(images_))]
    save_images_in_path(images_, names_, images_path)
