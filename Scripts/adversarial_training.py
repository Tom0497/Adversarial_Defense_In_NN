import matplotlib.pyplot as plt
import models_and_utils as mm
import numpy as np
import tensorflow as tf
from imagenetData import ImageNetData, labels_to_one_hot

tf.compat.v1.enable_eager_execution()


batch_size = 64
epochs = 5
images_per_class = 500
batch_number = int(images_per_class / batch_size)
dropout_rate = .2
classes = [96, 950, 530]  # ,447, 530, 592, 950, 96]
n_classes = len(classes)
imageNet = ImageNetData(classes, images_per_class=500,
                        batch_size=batch_size, validation_proportion=0.4)

model = mm.define_model(n_classes, use_pre_trained=True)
model.load_weights('best_model_val_loss.hdf5')


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


def generate_adversarials(examples, labels, number_of_examples=None, image_list=None, epsilon=None, use_step_ll=False):
    while True:
        x = []
        original_x = []
        y = []

        if image_list is None:
            image_list = list(range(len(labels)))
            np.random.shuffle(image_list)

        if number_of_examples is None:
            number_of_examples = len(image_list)

        for example in range(number_of_examples):
            n = image_list[example]
            original_x.append(n)

            label = labels[n]
            image = examples[n]

            if use_step_ll:
                perturbations = adversarial_step_ll(image.reshape(1, 224, 224, 3)).numpy()
            else:
                perturbations = adversarial_pattern(image.reshape((1, 224, 224, 3)), label).numpy()

            if epsilon is None:
                epsilon = tf.abs(tf.random.truncated_normal([1, 1], mean=0, stddev=1)).numpy()[0][0]

            adversarial = image + perturbations * epsilon

            x.append(adversarial)
            y.append(labels[n])

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


x_train, y_train = imageNet.get_train_set()

x_adversarial_train, x_original_train, y_adversarial_train = next(generate_adversarials(examples=x_train,
                                                                                        labels=y_train,
                                                                                        number_of_examples=300))


x_test, y_test = imageNet.get_train_set()

number_of_adv_examples = 100
random_images = list(range(len(y_test)))
np.random.shuffle(random_images)
random_images = random_images[:number_of_adv_examples]

epsilons = np.linspace(0, 2, num=21)

x_adversarial_test_epsilons = []
y_adversarial_test_epsilons = []
adv_test_accu_before_epsilons = []


for epsilon in epsilons:
    if epsilon != 0:
        x_adversarial_test, _, y_adversarial_test = next(generate_adversarials(examples=x_test, labels=y_test,
                                                                               image_list=random_images,
                                                                               epsilon=epsilon))
        x_adversarial_test_epsilons.append(x_adversarial_test)
        y_adversarial_test_epsilons.append(y_adversarial_test)
        accuracy = model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0)[1]
    else:
        accuracy = mm.to_test_model(model, imageNet)
    adv_test_accu_before_epsilons.append(accuracy)
    print(f"Accuracy base, epsilon {epsilon}: {accuracy}")


x_val, y_val = imageNet.get_validation_set()
validation_adv_clean_proportion = 0.5
val_adv_number = int(len(y_val) * validation_adv_clean_proportion)
x_adversarial_val, x_original_val, y_adversarial_val = next(generate_adversarials(examples=x_val, labels=y_val,
                                                                                  number_of_examples=val_adv_number))

x_val_final = []
y_val_final = []
for index, example in enumerate(x_val):
    if index not in x_original_val:
        x_val_final.append(example)
        y_val_final.append(y_val[index])

ensemble_x_val = np.r_[np.asarray(x_val_final), x_adversarial_val]
ensemble_y_val = np.r_[np.asarray(y_val_final), y_adversarial_val]

model.fit(x_adversarial_train, y_adversarial_train, batch_size=batch_size, epochs=epochs,
          validation_data=(ensemble_x_val, ensemble_y_val))

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

fig = plt.figure(figsize=(30, 30))
columns = 10
rows = 10
count = 1
saved_images = []
for _ in range(10):
    i = np.random.randint(list(x_adversarial_test_epsilons[0].shape)[0])
    while i in saved_images:
        i = np.random.randint(list(x_adversarial_test_epsilons[0].shape)[0])
    for j in range(0, 2 * rows, 2):
        x = x_adversarial_test_epsilons[j][i] * imageNet.std + imageNet.mean
        a_min = np.min(x)
        a_max = np.max(x)
        a_scaled = (x - a_min) / (a_max - a_min)
        fig.add_subplot(rows, columns, count)
        count += 1

        figs = plt.imshow(a_scaled, interpolation="nearest")
        plt.axis('off')
        figs.axes.get_xaxis().set_visible(False)
        figs.axes.get_yaxis().set_visible(False)
plt.show()
