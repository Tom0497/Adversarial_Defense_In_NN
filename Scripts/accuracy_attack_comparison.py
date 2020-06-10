"""
accuracy_attack_comparison.py: script that generates report's Figures 1 and 2. Script 'adversarial_generation.py' must
be run before this, since the csv files generated from it will be used
"""
import pandas as pd
import matplotlib.pyplot as plt

# Figure 1
fg = pd.read_csv('fg.csv')
fgsm = pd.read_csv('fgsm.csv')
rfgs = pd.read_csv('rfgs.csv')
step_ll = pd.read_csv('step_ll.csv')
plt.plot(fg['epsilons'], fg['accuracy'], 'o-', label='FGM')
plt.plot(fgsm['epsilons'], fgsm['accuracy'], 'o-', label='FGSM')
plt.plot(rfgs['epsilons'], rfgs['accuracy'], 'o-', label='R-FGSM')
plt.plot(step_ll['epsilons'], step_ll['accuracy'], 'o-', label='step l.l.')
plt.title('$\epsilon$ vs accuracy top-1')
plt.xlabel('$\epsilon$')
plt.ylabel('accuracy top-1')
plt.legend()
plt.savefig('accuracy_top1.eps', format='eps')
plt.show()

# Figure 2
plt.plot(fg['epsilons'], fg['accuracy5'], 'o-', label='FGM')
plt.plot(fgsm['epsilons'], fgsm['accuracy5'], 'o-', label='FGSM')
plt.plot(rfgs['epsilons'], rfgs['accuracy5'], 'o-', label='R-FGSM')
plt.plot(step_ll['epsilons'], step_ll['accuracy5'], 'o-', label='step l.l.')
plt.title('$\epsilon$ vs accuracy top-5')
plt.xlabel('$\epsilon$')
plt.ylabel('accuracy top-5')
plt.legend()
plt.savefig('accuracy_top5.eps', format='eps')
plt.show()