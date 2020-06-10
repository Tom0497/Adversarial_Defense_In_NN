import os
import sys

from Scripts.data_extractor import download_images_by_int_label

classes_file = r"/imagenet_class_index.json"
urls_file = r"/fall11_urls.txt"
current_directory = os.getcwd()
dict_path = os.path.dirname(current_directory) + r"/image_metadata" + classes_file
urls_path = os.path.dirname(current_directory) + r"/image_metadata" + urls_file
images_path = os.path.dirname(current_directory) + r"/images"
urls_folder_path = os.path.dirname(current_directory) + r"/image_metadata/urls"

if __name__ == "__main__":
    inputs = sys.argv

    img_per_class = int(inputs[1])
    start_class = int(inputs[2])
    stop_class = int(inputs[3])

    for i in range(start_class, stop_class + 1):
        download_images_by_int_label(i, images_path,
                                     urls_path,
                                     urls_folder_path,
                                     dict_path,
                                     download_limit=img_per_class,
                                     starting_url=1)
