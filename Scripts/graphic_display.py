from pickle import load
from models_and_utils import get_opt_and_eps, summary_of_register
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.reset_defaults()
sns.set(context='paper', style='ticks', font_scale=1.5)

if __name__ == "__main__":
    a_key = ['model_rmsprop_0001', 'model_adam_0001', 'model_sgd_0005', 'model_adagrad_00005']
    best_histories = {}
    for name in a_key:
        dict_file = open('trainHistoryLogs/{}_hist.pkl'.format(name), 'rb')
        history = load(dict_file)
        dict_file.close()
        best_histories.update(history)
    # summary_of_register(best_histories)

    """
    dict_file = open('trainHistoryLogs/train_models_hist.pkl', 'rb')
    histories = load(dict_file)
    dict_file.close()
    # These are the best model from the Grid Search
    a_key = ['model_rmsprop_0001', 'model_adam_0001', 'model_sgd_0005', 'model_adagrad_00005']
    sub_dict = {k: histories[k] for k in a_key}
    summary_of_register(sub_dict)

    lrs = []
    training_times = {}
    for model_name, _ in histories.items():
        opt, lr = get_opt_and_eps(model_name)
        if lr not in lrs:
            lrs.append(lr)
        training_times[opt] = []

    for model_name, history in histories.items():
        opt, _ = get_opt_and_eps(model_name)
        training_times[opt].append(history['training_time'])

    plt.figure()
    for opt, train_times in training_times.items():
        plt.plot(lrs, train_times, label=opt, marker='o', alpha=.9)
    plt.xlabel('Learning rate')
    plt.ylabel('Training time (Sec)')
    plt.legend()
    plt.title('Time to train vs learning rate, by optimizer')
    plt.xscale('log')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    best_models = ['model_rmsprop_0001', 'model_adam_0001', 'model_sgd_0005', 'model_adagrad_00005']
    best_histories = {k: histories[k] for k in best_models}

    plt.figure()
    for model_name, history in best_histories.items():
        opt, _ = get_opt_and_eps(model_name)
        plt.plot(np.cumsum(history['time_history']), history['val_acc'], label=opt, marker='o', alpha=.9)
    plt.xlabel('Time (Sec)')
    plt.ylabel('Validation accuracy')
    plt.legend()
    plt.title('Validation accuracy over training time, by optimizer')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    """
