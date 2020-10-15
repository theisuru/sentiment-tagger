import math

import pandas as pd
from scipy import stats

classifier_accuracy = 0.88
data_file = '../../../corpus/analyzed/comments_tagged_remove.csv'


def get_label_count(accuracy, data_file):
    comments_df = pd.read_csv(data_file, ';')
    counts = comments_df['label'].value_counts()
    negative_count = counts['NEGATIVE']
    positive_count = counts['POSITIVE']
    total_count = negative_count + positive_count

    p_negative = negative_count / total_count
    p_positive = positive_count / total_count

    acc_null = math.pow(p_positive, 2) + math.pow(p_negative, 2)
    p_value = stats.binom_test(round(total_count * accuracy), total_count, acc_null, alternative='greater')

    print('Null hypothesis: ' + str(acc_null))
    print('P-Value: ' + str(p_value))


get_label_count(classifier_accuracy, data_file)
