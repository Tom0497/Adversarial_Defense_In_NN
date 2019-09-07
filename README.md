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
