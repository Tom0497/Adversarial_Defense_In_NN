from pickle import load
from models_and_utils import summary_of_register

if __name__ == "__main__":
    dict_file = open('trainHistoryLogs/train_models_hist.pkl', 'rb')
    histories = load(dict_file)
    dict_file.close()

    a_key = ['model_rmsprop_0001', 'model_adam_0001', 'model_sgd_0005']
    sub_dict = {k: histories[k] for k in a_key}

    summary_of_register(sub_dict)
