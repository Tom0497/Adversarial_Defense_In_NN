import pandas as pd
import os
from PIL import Image
import urllib.request
import urllib.error
import http.client
import json
import ssl
import sys

classes_file = r"/imagenet_class_index.json"
urls_file = r"/fall11_urls.txt"
current_directory = os.getcwd()
dict_path_ = os.path.dirname(current_directory) + r"/image_metadata" + classes_file
urls_path = os.path.dirname(current_directory) + r"/image_metadata" + urls_file
images_path = os.path.dirname(current_directory) + r"/images"
urls_folder_path = os.path.dirname(current_directory) + r"/image_metadata/urls"


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
        image = Image.open(urllib.request.urlopen(url, timeout=10))
        image.verify()
        image = Image.open(urllib.request.urlopen(url, timeout=10))
        return image
    except (urllib.error.HTTPError,
            urllib.error.URLError,
            IOError,
            http.client.HTTPException,
            SyntaxError,
            ssl.CertificateError) as e:
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
    try:
        image.save(os.path.join(path, image_name), 'JPEG')
    except IOError as e:
        print(e)


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


def filter_urls_by_wnid(data, wnid):
    """
    Returns list of urls for given wnid.
    :param data:        DataFrame. Columns:
                            * 'wnid': Contains wnid
                            * 'url':  Contains url
    :param wnid:        Desired wnid to filter urls
    :return:            List of urls for give wnid.
    """
    data_filtered = data[data['wnid'] == wnid]
    urls_filtered = data_filtered["url"].values
    return urls_filtered


def get_urls_by_wnid(wnid, path):
    """
    Returns all urls associated to a specific wnid.

    :param wnid:        Desired wnid to filter urls
    :param path:        Path to the .txt file that contains the raw version of the wnid and url data for the images
    :return:            List of urls associated to specified wnid.
    """
    url_list = []
    with open(path, encoding="latin-1") as file:
        for line in file:
            line_list = line.strip().split("\t")
            wnid_read = line_list[0].partition("_")[0]
            if wnid_read == wnid:
                url_list.append(line_list[1])
    return url_list


def generate_urls_file(wnid, raw_path, url_folder_path):
    """

    :param wnid:            Desired wnid to filter urls
    :param raw_path:        Path to the .txt file that contains the raw version of the wnid and url data for the images
    :param url_folder_path: Path to the folder in which url files will be saved
    :return:
    """
    url_list = get_urls_by_wnid(wnid, raw_path)
    if not os.path.isdir(url_folder_path):
        os.mkdir(url_folder_path)
    with open(url_folder_path + rf'/{wnid}.txt', 'w', encoding='latin-1') as f:
        for url in url_list:
            f.write("%s\n" % url)


def download_images_by_wnid(wnid, image_folder, raw_path, url_folder_path, json_path, download_limit=16,
                            starting_url=0):
    """
    Downloads all the images associated with the given wnid into the specified folder, to get the urls first checks if
    a the file of urls exists, if not, it creates it, then it begins to download the images up to a limit, also can be
    specified in which line of the urls file the download should start.

    :param json_path:           path to the file that associates a wnid code with the class's number
    :param wnid:                the wnid code of the images that'll be downloaded
    :param image_folder:        the destination folder where to download the images
    :param raw_path:            path to the file that contains all the urls
    :param url_folder_path:     path to the folder where the urls files should be
    :param download_limit:      maximum amount of images to download when the function is called
    :param starting_url:        the line in urls file where to start reading urls
    """
    url_file = url_folder_path + rf'/{wnid}.txt'
    if not os.path.isfile(url_file):
        generate_urls_file(wnid, raw_path, url_folder_path)

    number_label, label = get_labels_for_wnid(wnid, json_path)

    image_destination_path = image_folder + rf'/{number_label}_{label}'
    if not os.path.isdir(image_destination_path):
        os.mkdir(image_destination_path)

    with open(url_file, encoding="latin-1") as file:
        url_counter = 0
        download_counter = 0
        for line in file:
            if url_counter < starting_url:
                url_counter += 1
                continue
            image_name = f'{wnid}_{url_counter}.jpg'
            if download_counter < download_limit:
                if os.path.isfile(os.path.join(image_destination_path, image_name)):
                    url_counter += 1
                    continue
                image = get_image_from_url(line)
                if image is None:
                    url_counter += 1
                    continue
                save_image_in_path(image, image_name, path=image_destination_path)
                url_counter += 1
                download_counter += 1
            else:
                return


def download_images_by_label(label, image_folder, raw_path, url_folder_path, json_path, download_limit=16, starting_url=0):
    """
    Downloads all the images associated with the given label into the specified folder, to get the urls first checks if
    a the file of urls exists, if not, it creates it, then it begins to download the images up to a limit, also can be
    specified in which line of the urls file the download should start.

    :param label:               the label of the images that'll be downloaded
    :param json_path:           path to the file that associates a wnid code with the class's number
    :param image_folder:        the destination folder where to download the images
    :param raw_path:            path to the file that contains all the urls
    :param url_folder_path:     path to the folder where the urls files should be
    :param download_limit:      maximum amount of images to download when the function is called
    :param starting_url:        the line in urls file where to start reading urls
    """
    wnid = get_wnid_for_label(label, json_path)
    download_images_by_wnid(wnid, image_folder, raw_path, url_folder_path, json_path, download_limit, starting_url)


def download_images_by_int_label(int_label, image_folder, raw_path, url_folder_path, json_path, download_limit=16, starting_url=0):
    """
    Downloads all the images associated with the given integer label into the specified folder, to get the urls first
    checks if a the file of urls exists, if not, it creates it, then it begins to download the images up to a limit,
    also can be specified in which line of the urls file the download should start.

    :param int_label:           the number of the class of the images that'll be downloaded
    :param json_path:           path to the file that associates a wnid code with the class's number
    :param image_folder:        the destination folder where to download the images
    :param raw_path:            path to the file that contains all the urls
    :param url_folder_path:     path to the folder where the urls files should be
    :param download_limit:      maximum amount of images to download when the function is called
    :param starting_url:        the line in urls file where to start reading urls
    """
    wnid = get_wnid_for_int_label(int_label, json_path)
    download_images_by_wnid(wnid, image_folder, raw_path, url_folder_path, json_path, download_limit, starting_url)


def get_wnid_for_int_label(int_label, json_path):
    """
    It gets the wnid code for an integer label from the file specified in json_path

    :param int_label:       the number of a class
    :param json_path:       the path to the file that has the wnid-label relation
    :return:                the wnid associated with the int label
    """
    key = str(int_label)
    with open(json_path) as json_file:
        data = json.load(json_file)
        try:
            return data[key][0]
        except KeyError:
            return


def get_wnid_for_label(label, json_path):
    """
    It gets the wnid code for a label from the file specified in json_path

    :param label:           the label of a class
    :param json_path:       the path to the file that has the wnid-label relation
    :return:                the wnid associated with the label
    """
    with open(json_path) as json_file:
        data = json.load(json_file)
        for key in data.keys():
            value = data[key]
            if value[1] == label:
                return value[0]


def get_labels_for_wnid(wnid, json_path):
    """
    It gets both the integer and the name associated with a wnid from the file in json_path

    :param wnid:            the wnid of a class
    :param json_path:       the path to the file that has the wnid-label relation
    :return:                a tuple with both the integer and the name of a class
    """
    with open(json_path) as json_file:
        data = json.load(json_file)
        for key in data.keys():
            value = data[key]
            if value[0] == wnid:
                return key, value[1]


if __name__ == "__main__":
    inputs = sys.argv

    img_per_class = int(inputs[1])
    start_class = int(inputs[2])
    stop_class = int(inputs[3])

    for i in range(start_class, stop_class + 1):
        download_images_by_int_label(i, images_path,
                                     urls_path,
                                     urls_folder_path,
                                     dict_path_,
                                     download_limit=img_per_class,
                                     starting_url=1)
