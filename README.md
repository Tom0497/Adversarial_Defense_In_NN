# Adversarial Defense in Neural Networks
***Project for the Computational Intelligence course (EL-4106), spring semester, 2019.***
### Authors:    
**R. D. Becerra** <br>
Electrical Engineering Student <br> Physical and Mathematical Sciences Faculty <br> Universidad de Chile <br>
raimundo.becerra@ug.uchile.cl <br>
<br>
**T. R. Saldivia** <br>
Electrical Engineering Student <br> Physical and Mathematical Sciences Faculty <br> Universidad de Chile <br>
tomas.saldivia@ug.uchile.cl <br>
<div style="text-align: justify">
The current repository will eventually contain all the scripts, codes, and resources necessary to successfully achieve the objective of this project.

## Description
A remarkable feature in human perception is the capacity for local generalization. It consists on the ability to recognize common qualities among objects that are similar to each other, or between the object itself when observed from different angles, luminosities, or sizes. This quality is typically valid for *Computer Vision* systems. When applied to *Neural Networks* that perform classification, it'd suppose that new data, similar to training examples, could be classified within the same class. However, this doesn't always happen. Such data are called *Adversarial Examples*. <br>

This proyect's objective is to generate *adversarial examples* of *Neural Networks* using two different methods of free choice, using data from [***ImageNet***](http://www.image-net.org/). After comparing both algorithms' performance, with the generated adversarial examples an adversarial defense algorithm must be implemented and it has to improve the robustness of a *Convolutional Neural Network* faced to this kind of attacks. <br>

Finally, given that adversarial examples are effective before different classifiers' architectures, and before models that have been trained with different databases but meet the same task, it is required to evaluate the performace of adversarial attacks of the *black box* kind to [***GoogleCloudVision***](https://cloud.google.com/vision/) (GCV) System or to [***Clarif AI***](https://www.clarifai.com/), where there's no access to the model being attacked. 

## Code Execution
### Requirements

First of all, it must be taken into consideration that this code was developed using ***PyCharm IDE*** and an enviroment running ***Python 3.7***. For must of the classification task, Tensorflow's API Keras was used, using the version 1.15 of Tensorflow. A library used for adversarial example generation was foolbox, which can be installed using the command `pip install foolbox`. Besides these, standard scientific libraries such as pandas, matplotlib, numpy, etc; were used.

### Settting the local repository

Before executing any piece of code, it's imperative to have the scripts in your computer, so the first thing to do is clone or download this repository, the former through the command `git clone https://github.com/Tom0497/Adversarial_Defense_In_NN.git` in console or bash depending of the operative system. <br>

Once this is done, you must go to the repository's folder and create two other folders with these exacts names: `images` and `image_metadata`. In the last, two files must be placed, one is a .txt file that contains all the urls of the images from the last competition version of the ImageNet dataset, this can be obtained from [ImageNet_URLs](http://image-net.org/download-imageurls), it must be mentioned that along with the URLs, comes the WnID code in order to recover the class associated with the image. The second file corresponds to the class index file, which maps the classes of the dataset with its respective WnID code, this can be obtained from [here](https://github.com/USCDataScience/dl4j-kerasimport-examples/blob/master/dl4j-import-example/data/imagenet_class_index.json). Once they both are downloaded, they must be saved to the folder `image_metadata`. <br>

The other folder must be created empty, and it'll be the location of where the images will be downloaded. So, once you checked the installation of all necessary libraries, and have created the folders and added the files, the directory is ready to be used. As a side note an IDE like PyCharm can be used to open the code and check if anything is left to do, like installing a library or something else.

### Obtaining Images

The first thing needed for any machine learning problem is the data, and in this particular case we've developed a sort of API that can obtain images from the ImageNet dataset and store them in your computer, to do so, follow these steps.
- Go to the `...\directory\Scripts` folder then execute the following command:

- `python image_getter.py num_per_class start_class stop_class`

Where *num_per_class* indicates how many images from each class are going to be downloaded, it must be said that this task can take some considerable time so first you should try with small values, no higher than 5. This script will handle both the reading of the URLs as the creation of new sub-files of URLs separated by WnID code, then, it will read from these files and then get the images until it fills the number of images specified. All the images are saved within the image folder in separate folders. The other two parameters are used to indicate the range of classes to execute the described task. For all the classes the values must be `start_class = 0` and `stop_class = 999`.

### Generating and visualizing the adversarial examples

For generating and visualizing adversarial examples follow these instructions:
- Go to the `...\directory\Scripts` folder then execute the following command:
- `python attackImplemented.py num_class epsilon grad_attack`

Where *num_class* indicates from which of the classes existing in the image folder generate adversarial examples. Then, it must be taken into consideration that all gradient based attacks have an hyperparameter epsilon, which in this case controls how distorted is the resulting image after the attack. You can try with several values, we recommend some value between 10^-2 y 1. Then the last argument *grad_attack* indicates which of the implemented gradient based attack to use, it can be `fg` for Fast Gradient Method, `fgsm` for Fast Gradient Sign Method or `rfgs` for Random Fast Gradient Sign Method. So make sure to fill all the args and check if the specified folder exists and contains images.

### Evaluating the model

In order to analize the model's performance a script was created, in it, the accuracy of the model is computed for a set of adversarial examples and a value of epsilon. In order to check this feature, follow these steps:
- Go to the `...\directory\Scripts` folder, the executes the next line:
- `python model_evaluation.py max_classes epsilon attack_type`

Where *max_classes* indicates how many diferent sets must be used to compute the accuracy, knowing that exactly one image per set will be taken. Then *epsilon* is, as described before, the hyperparameter of the gradient based attacks. Last, *attack_type* indicates which of the three already mentioned attacks will be used for generating the adversarial examples. When the scripy execution is done, you should see both top-1 and top-5 accuracy of the model over the generated adversarial examples.
