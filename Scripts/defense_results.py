"""
defense_results.py: a script for visualizing main results of adversarial training
"""

from pickle import load

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.reset_defaults()
sns.set(context='paper', style='ticks', font_scale=1.5)

if __name__ == '__main__':
    # attacks considered
    attacks = ['fgsm', 'fg', 'rfgs', 'step_ll']

    # load of defense training history
    hist = {}
    for att in attacks:
        with open(f'../logs/history/defense/adv_def_hist_{att}.pkl', 'rb') as f:
            hist[att] = load(f)

    # epsilons used in adversarial training
    epsilons = np.linspace(0, 3, num=13)

    # plot of the results in terms of accuracy
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey='row', dpi=300)
    for att in attacks:
        axs[0].plot(epsilons, hist[att]['acc_no_def'], linewidth=3, alpha=.9)
        axs[0].set(title='Sin defensa adversaria', ylabel='Accuracy', xlabel='epsilon')
        axs[1].plot(epsilons, hist[att]['acc_def'], label=att, linewidth=3, alpha=.9)
        axs[1].set(title='Con defensa adversaria', xlabel='epsilon')
        axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.suptitle('Accuracy en conjunto de test', y=0.995)
    fig.show()
