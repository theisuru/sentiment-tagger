import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import seaborn as sns




sns.set_style("whitegrid")
n_groups = 3

medagoda_results = (60, 58, 56)
our_results = (77.69, 76.54, 86.17)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.3

opacity = 1
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, medagoda_results, bar_width,
                alpha=opacity, color='#791D0A', error_kw=error_config,
                label='Medagoda')

rects2 = ax.bar(index + bar_width, our_results, bar_width,
                alpha=opacity, color='#0A4079', error_kw=error_config,
                label='Our')

ax.set_xlabel('Classifier')
ax.set_ylabel('Accuracy')
ax.set_title('Classifier Accuracy')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('Naive Bayes', 'Decision Tree', 'SVM', 'RNN LSTM'))
ax.legend()

fig.tight_layout()
plt.show()