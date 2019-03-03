import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import seaborn as sns




sns.set_style("whitegrid")
n_groups = 4

naive_bayes = (78, 84, 69, 75)
decision_tree = (77, 76, 76, 77)
svm = (86, 92, 79, 85)
rnn = (86, 89, 85, 86)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2

opacity = 0.8
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, naive_bayes, bar_width,
                alpha=opacity, color='#791E0A', error_kw=error_config,
                label='Naive Bayes')

rects2 = ax.bar(index + bar_width, decision_tree, bar_width,
                alpha=opacity, color='#0F790A', error_kw=error_config,
                label='Decision Tree')


rects3 = ax.bar(index + 2 * bar_width, svm, bar_width,
                alpha=opacity, color='#0A2B79', error_kw=error_config,
                label='SVM')


rects4 = ax.bar(index + 3 * bar_width, rnn, bar_width,
                alpha=opacity, color='#790A6D', error_kw=error_config,
                label='RNN LSTM')

ax.set_xlabel('Metric')
ax.set_ylabel('Value')
ax.set_title('Performance of Classifiers')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(('Accuracy', 'Precision', 'Recall', 'F1 Score'))
ax.legend()

fig.tight_layout()
plt.show()