import adversarial_utils as au
import matplotlib.pyplot as plt
import models_and_utils as mm
import numpy as np
import tensorflow as tf
from imagenetData import ImageNetData
from tensorflow.python.keras.callbacks import ModelCheckpoint

tf.compat.v1.enable_eager_execution()

if __name__ == "__main__":
    epochs = 50
    batch_size = 64
    dropout_rate = .2
    images_per_class = 500
    number_of_adv_examples = 100
    batch_number = int(images_per_class / batch_size)

    classes = [96, 950, 530]  # Available classes : [447, 530, 592, 950, 96]
    n_classes = len(classes)

    imageNet = ImageNetData(classes, images_per_class=images_per_class,
                            batch_size=batch_size, validation_proportion=0.4)

    model = mm.define_model(n_classes, use_pre_trained=True, learning_rate=0.0005)
    model.load_weights('best_model_val_loss.hdf5')

    x_train, y_train = imageNet.get_train_set()
    x_test, y_test = imageNet.get_test_set()
    x_val, y_val = imageNet.get_validation_set()

    x_adversarial_train, x_original_train, y_adversarial_train = next(au.generate_adversarial(model=model,
                                                                                              examples=x_train,
                                                                                              labels=y_train,
                                                                                              num_classes=n_classes))

    random_images = list(range(len(y_test)))
    np.random.shuffle(random_images)
    random_images = random_images[:number_of_adv_examples]

    epsilons = np.linspace(0, 2, num=21)

    x_adversarial_test_epsilons = []
    y_adversarial_test_epsilons = []
    adv_test_accu_before_epsilons = []

    for epsilon in epsilons:
        if epsilon != 0:
            x_adversarial_test, _, y_adversarial_test = next(au.generate_adversarial(model=model,
                                                                                     examples=x_test,
                                                                                     labels=y_test,
                                                                                     num_classes=n_classes,
                                                                                     image_list=random_images,
                                                                                     epsilon=epsilon))
            x_adversarial_test_epsilons.append(x_adversarial_test)
            y_adversarial_test_epsilons.append(y_adversarial_test)
            accuracy = model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0)[1]
        else:
            accuracy = mm.to_test_model(model, imageNet)
        adv_test_accu_before_epsilons.append(accuracy)
        print("Accuracy base, epsilon {0:.1f}: {1:.3f}".format(epsilon, accuracy))

    validation_adv_clean_proportion = 0.5
    val_adv_number = int(len(y_val) * validation_adv_clean_proportion)
    x_adversarial_val, x_original_val, y_adversarial_val = next(au.generate_adversarial(model=model,
                                                                                        examples=x_val,
                                                                                        labels=y_val,
                                                                                        num_classes=n_classes,
                                                                                        number_of_examples=
                                                                                        val_adv_number))

    x_val_clean = []
    y_val_clean = []
    for index, example in enumerate(x_val):
        if index not in x_original_val:
            x_val_clean.append(example)
            y_val_clean.append(y_val[index])

    x_val_final = np.r_[np.asarray(x_val_clean), x_adversarial_val]
    y_val_final = np.r_[np.asarray(y_val_clean), y_adversarial_val]

    checkpoint = ModelCheckpoint('best_adv_model_val_loss.hdf5', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min', save_weights_only=True)

    history = model.fit(x_adversarial_train, y_adversarial_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(x_val_final, y_val_final))

    model.load_weights('best_adv_model_val_loss.hdf5')

    adv_test_accu_after_epsilons = []
    for epsilon_index in range(len(epsilons)):
        if epsilon_index != 0:
            accuracy = model.evaluate(x=x_adversarial_test_epsilons[epsilon_index - 1],
                                      y=y_adversarial_test_epsilons[epsilon_index - 1], verbose=0)[1]
        else:
            accuracy = mm.to_test_model(model, imageNet)
        adv_test_accu_after_epsilons.append(accuracy)
        print("Accuracy fitted, epsilon {0:.1f}: {1:.3f}".format(epsilons[epsilon_index], accuracy))

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

    mm.plot_learning_curves(history, epochs)

    plt.show()
