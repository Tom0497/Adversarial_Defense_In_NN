import matplotlib.pyplot as plt
import models_and_utils as mm
import numpy as np
import tensorflow as tf
from imagenetData import ImageNetData, labels_to_one_hot

tf.compat.v1.enable_eager_execution()

batch_size = 64
epochs = 30
images_per_class = 500
batch_number = int(images_per_class / batch_size)
dropout_rate = .2
classes = [96, 950, 530]  # ,447, 530, 592, 950, 96]
n_classes = len(classes)
imageNet = ImageNetData(classes, images_per_class=500,
                        batch_size=batch_size, validation_proportion=0.4)

model = mm.define_model(n_classes, use_pre_trained=True)
model.load_weights('best_pretrained_weights.hdf5')


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
        y_ll = model(image).numpy().argmin()
        y_ll = labels_to_one_hot([y_ll], n_classes)[0]
        loss = tf.keras.losses.MSE(y_ll, prediction)

    gradient = tape.gradient(loss, image)

    signed_grad = -1 * tf.sign(gradient)

    return signed_grad


def generate_adversarials(number_of_examples, epsilon=None, use_step_ll=False):
    while True:
        x = []
        original_x = []
        y = []

        x_train, y_train = imageNet.get_train_set()

        ns = list(range(len(y_train)))
        np.random.shuffle(ns)
        for example in range(number_of_examples):

            n = ns[example]
            original_x.append(n)

            label = y_train[n]
            image = x_train[n]

            if use_step_ll:
                perturbations = adversarial_step_ll(image.reshape(1, 224, 224, 3)).numpy()
            else:
                perturbations = adversarial_pattern(image.reshape((1, 224, 224, 3)), label).numpy()

            if epsilon is None:
                epsilon = tf.abs(tf.random.truncated_normal([1, 1], mean=0, stddev=1)).numpy()[0][0]

            adversarial = image + perturbations * epsilon

            x.append(adversarial)
            y.append(y_train[n])

        x = np.asarray(x).reshape((number_of_examples, 224, 224, 3))
        original_x = np.asarray(original_x)
        y = np.asarray(y)

        yield x, original_x, y


def generate_adversarials_by_image_list(image_list, epsilon=None, use_step_ll=False):
    while True:
        x = []
        y = []

        x_train, y_train = imageNet.get_train_set()

        for example in range(len(image_list)):
            n = image_list[example]

            label = y_train[n]
            image = x_train[n]

            if use_step_ll:
                perturbations = adversarial_step_ll(image.reshape((1, 224, 224, 3))).numpy()
            else:
                perturbations = adversarial_pattern(image.reshape((1, 224, 224, 3)), label).numpy()

            if epsilon is None:
                epsilon = tf.abs(tf.truncated_normal([1, 1], mean=0, stddev=1)).numpy()[0][0]

            adversarial = image + perturbations * epsilon

            x.append(adversarial)
            y.append(y_train[n])

        x = np.asarray(x).reshape((len(image_list), 224, 224, 3))
        y = np.asarray(y)

        yield x, y


x_adversarial_train, x_original_train, y_adversarial_train = next(generate_adversarials(500, use_step_ll=True))

x_train, y_train = imageNet.get_train_set()


number_of_adv_examples = 100
random_images = list(range(len(y_train)))
np.random.shuffle(random_images)
random_images = random_images[:number_of_adv_examples]

epsilons = np.linspace(0, 2, num=20)

x_adversarial_test_epsilons = []
y_adversarial_test_epsilons = []
adv_test_accu_before_epsilons = []

for epsilon in epsilons:
    if epsilon != 0:
        x_adversarial_test, y_adversarial_test = next(generate_adversarials_by_image_list(random_images, epsilon=epsilon))
        x_adversarial_test_epsilons.append(x_adversarial_test)
        y_adversarial_test_epsilons.append(y_adversarial_test)
        accuracy = model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0)[1]
    else:
        accuracy = mm.to_test_model(model, imageNet)
    adv_test_accu_before_epsilons.append(accuracy)
    print(f"Accuracy base, epsilon {epsilon}: {accuracy}")


model.fit(x_adversarial_train, y_adversarial_train, batch_size=batch_size, epochs=epochs,
          validation_data=imageNet.get_validation_set())

adv_test_accu_after_epsilons = []
for epsilon_index in range(len(epsilons)):
    if epsilon_index != 0:
        accuracy = model.evaluate(x=x_adversarial_test_epsilons[epsilon_index-1],
                                  y=y_adversarial_test_epsilons[epsilon_index-1], verbose=0)[1]
    else:
        accuracy = mm.to_test_model(model, imageNet)
    adv_test_accu_after_epsilons.append(accuracy)
    print(f"Accuracy fitted, epsilon {epsilons[epsilon_index]}: {accuracy}")

plt.plot(epsilons, adv_test_accu_before_epsilons, 'ro', label='Before adv. training')
plt.plot(epsilons, adv_test_accu_after_epsilons, 'bo', label='After adv. training')
plt.legend()
plt.show()
