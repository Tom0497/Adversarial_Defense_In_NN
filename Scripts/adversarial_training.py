import numpy as np
import tensorflow as tf
import random
from imagenetData import ImageNetData, labels_to_one_hot
from tensorflow.python.keras.models import load_model
import models_and_utils as mm
from tensorflow.python.keras.utils import to_categorical
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()


batch_size = 64
epochs = 5
images_per_class = 500
batch_number = int(images_per_class / batch_size)
dropout_rate = .2
classes = [96, 950, 530]  # ,447, 530, 592, 950, 96]
n_classes = len(classes)
imageNet = ImageNetData(classes, images_per_class=500,
                        batch_size=batch_size, validation_proportion=0.4, augment_data=True)

model = mm.define_model(n_classes, own_num=1)
model.load_weights('best_model_1.hdf5')

def adversarial_pattern(image, label):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)

    gradient = tape.gradient(loss, image)

    signed_grad = tf.sign(gradient)

    return signed_grad

def adversarial_step_ll(image):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        y_ll = model.predict(image.copy()[np.newaxis, :]).argmin()
        y_ll = labels_to_one_hot([y_ll], n_classes)[0]
        loss = tf.keras.losses.MSE(y_ll, prediction)

    gradient = tape.gradient(loss, image)

    signed_grad = -1 * tf.sign(gradient)

    return signed_grad


def generate_adversarials(number_of_examples, epsilon=None):
    while True:
        x = []
        original_x = []
        y = []
        n = -1

        x_train, y_train = imageNet.get_train_set()

        for example in range(number_of_examples):

            while n not in original_x:
                n = random.randint(0, len(y_train))
                original_x.append(n)

            label = y_train[n]
            image = x_train[n]

            perturbations = adversarial_pattern(image.reshape((1, 224, 224, 3)), label).numpy()

            if epsilon is None:
                epsilon = tf.abs(tf.truncated_normal([1, 1], mean=0, stddev=8)).numpy()[0][0]

            adversarial = image + perturbations * epsilon

            x.append(adversarial)
            y.append(y_train[n])

        x = np.asarray(x).reshape((number_of_examples, 224, 224, 3))
        original_x = np.asarray(original_x)
        y = np.asarray(y)

        yield x, original_x, y


x_adversarial_train, x_original_train, y_adversarial_train = next(generate_adversarials(300))

x_adversarial_test_01, x_original_test_01, y_adversarial_test_01 = next(generate_adversarials(100, epsilon=0.1))
x_adversarial_test_1, x_original_test_1, y_adversarial_test_1 = next(generate_adversarials(100, epsilon=1))
x_adversarial_test_3, x_original_test_3, y_adversarial_test_3 = next(generate_adversarials(100, epsilon=3))
x_adversarial_test_5, x_original_test_5, y_adversarial_test_5 = next(generate_adversarials(100, epsilon=5))
x_adversarial_test_8, x_original_test_8, y_adversarial_test_8 = next(generate_adversarials(100, epsilon=8))

test_accuracy_before = mm.to_test_model(model, imageNet)
adv_test_accu_before_01 = model.evaluate(x=x_adversarial_test_01, y=y_adversarial_test_01, verbose=0)[1]
adv_test_accu_before_1 = model.evaluate(x=x_adversarial_test_1, y=y_adversarial_test_1, verbose=0)[1]
adv_test_accu_before_3 = model.evaluate(x=x_adversarial_test_3, y=y_adversarial_test_3, verbose=0)[1]
adv_test_accu_before_5 = model.evaluate(x=x_adversarial_test_5, y=y_adversarial_test_5, verbose=0)[1]
adv_test_accu_before_8 = model.evaluate(x=x_adversarial_test_8, y=y_adversarial_test_8, verbose=0)[1]

print("Accuracy base, ejemplos normales:", test_accuracy_before)
print("Accuracy base, ejemplos adversarios, epsilon 0.1:", adv_test_accu_before_01)
print("Accuracy base, ejemplos adversarios, epsilon 1:", adv_test_accu_before_1)
print("Accuracy base, ejemplos adversarios, epsilon 3:", adv_test_accu_before_3)
print("Accuracy base, ejemplos adversarios, epsilon 5:", adv_test_accu_before_5)
print("Accuracy base, ejemplos adversarios, epsilon 8:", adv_test_accu_before_8)

model.fit(x_adversarial_train, y_adversarial_train, batch_size=batch_size, epochs=epochs,
          validation_data=imageNet.get_validation_set())

adv_test_accu_after_01 = model.evaluate(x=x_adversarial_test_01, y=y_adversarial_test_01, verbose=0)[1]
adv_test_accu_after_1 = model.evaluate(x=x_adversarial_test_1, y=y_adversarial_test_1, verbose=0)[1]
adv_test_accu_after_3 = model.evaluate(x=x_adversarial_test_3, y=y_adversarial_test_3, verbose=0)[1]
adv_test_accu_after_5 = model.evaluate(x=x_adversarial_test_5, y=y_adversarial_test_5, verbose=0)[1]
adv_test_accu_after_8 = model.evaluate(x=x_adversarial_test_8, y=y_adversarial_test_8, verbose=0)[1]
test_accuracy_after = mm.to_test_model(model, imageNet)

print("Accuracy fitted, ejemplos normales:", test_accuracy_after)
print("Accuracy fitted, ejemplos adversarios, epsilon 0.1:", adv_test_accu_after_01)
print("Accuracy fitted, ejemplos adversarios, epsilon 1:", adv_test_accu_after_1)
print("Accuracy fitted, ejemplos adversarios, epsilon 3:", adv_test_accu_after_3)
print("Accuracy fitted, ejemplos adversarios, epsilon 5:", adv_test_accu_after_5)
print("Accuracy fitted, ejemplos adversarios, epsilon 8:", adv_test_accu_after_8)

adv_test_accuracy_before = [adv_test_accu_before_01, adv_test_accu_before_1, adv_test_accu_before_3,
                            adv_test_accu_before_5, adv_test_accu_before_8]
adv_test_accuracy_after = [adv_test_accu_after_01, adv_test_accu_after_1, adv_test_accu_after_3,
                           adv_test_accu_after_5, adv_test_accu_after_8]
epsilons = [0, 0.1, 1, 3, 5, 8]

plt.plot(epsilons, test_accuracy_before + adv_test_accuracy_before, 'ro', label='Before adv. training')
plt.plot(epsilons, test_accuracy_after + adv_test_accuracy_before, 'bo', label='After adv. training')
plt.legend()
plt.show()