import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

medagoda_vs_ours_fig = "../../../results/medagoda_vs_ours.png"
classifier_comparison_fig = "../../../results/classifier_comparison.png"


def main():
    compare_accuracy_medagoda_vs_ours()
    compare_classifiers()


def compare_accuracy_medagoda_vs_ours():
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
    plt.savefig(medagoda_vs_ours_fig, format="png")
    plt.show()


def compare_classifiers():
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
    plt.savefig(classifier_comparison_fig, format="png")
    plt.show()


main()
