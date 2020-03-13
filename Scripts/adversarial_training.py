import numpy as np
import tensorflow as tf
import random
from imagenetData import ImageNetData, labels_to_one_hot
from tensorflow.python.keras.models import load_model
import models_and_utils as mm
from tensorflow.python.keras.utils import to_categorical

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
        n_prev = -1
        n = n_prev

        for example in range(number_of_examples):
            x_train, y_train = imageNet.get_train_set()

            while n == n_prev:
                n = random.randint(0, len(y_train))

            label = y_train[n]
            image = x_train[n]

            perturbations = adversarial_pattern(image.reshape((1, 224, 224, 3)), label).numpy()

            if epsilon is None:
                epsilon = tf.abs(tf.truncated_normal(mean=0, stddev=8))

            adversarial = image + perturbations * epsilon

            x.append(adversarial)
            original_x.append(n)
            y.append(y_train[n])

        x = np.asarray(x).reshape((number_of_examples, 224, 224, 3))
        original_x = np.asarray(original_x).reshape((number_of_examples, 224, 224, 3))
        y = np.asarray(y)

        yield x, original_x, y


x_adversarial_train, x_original_train, y_adversarial_train = next(generate_adversarials(300))

x_adversarial_test, x_original_test, y_adversarial_test = next(generate_adversarials(50))

test_accuracy_before = mm.to_test_model(model, imageNet)
adv_test_accu_before = model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0)[1]

print("Accuracy base, ejemplos normales:", test_accuracy_before)
print("Accuracy base, ejemplos adversarios:", adv_test_accu_before)

model.fit(x_adversarial_train, y_adversarial_train, batch_size=batch_size, epochs=epochs,
          validation_data=imageNet.get_validation_set())

adv_test_accu_after = model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0)[1]
test_accuracy_after = mm.to_test_model(model, imageNet)

print("Accuracy fitted, ejemplos normales:", test_accuracy_after)
print("Accuracy fitted, ejemplos adversarios:", adv_test_accu_after)
