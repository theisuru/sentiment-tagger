# coding: utf-8

import SentimentCommons as Commons
import W2VVectorizer
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from prettytable import PrettyTable
from sklearn.svm import SVC

def main():
    start_time = time.time()
    run_cross_val()
    # run_holdout()
    end_time = time.time()
    print("Time taken for the process: " + str(end_time - start_time))
    return


def run_holdout():
    w2v_model_path = "../../../corpus/analyzed/saved_models/"
    comments = pd.read_csv("../../../corpus/analyzed/comments_tagged.csv", ";")
    train_data, test_data = train_test_split(comments, test_size=0.4, random_state=0)
    print("Processing dataset: " + str(train_data.columns.values))

    print("Extracting features with W2V count vectorizer")
    vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_path + "word2vec_model_skipgram_300_10", True)
    # vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_path + "word2vec_model_from_unlabeled_comments_all_200", False)
    fit_models(vectorizer, train_data["comment"], test_data["comment"], train_data["label"], test_data["label"])

    return


def run_cross_val():
    all_predictions = []
    w2v_model_path = "../../../corpus/analyzed/saved_models/"
    comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove_all_punc.csv", ";")
    pretty_table = PrettyTable(["Algorithm", "Accuracy", "Precision", "Recall", "F1_Score"])

    i = 1
    kf = KFold(n_splits=10)
    kf.get_n_splits(comments)
    for train_index, test_index in kf.split(comments):
        train_data_comments, test_data_comments = comments["comment"][train_index], comments["comment"][test_index]
        train_data_labels, test_data_labels = comments["label"][train_index], comments["label"][test_index]
        # vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_path + "word2vec_model_skipgram_300", False)
        vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_path + "word2vec_model_skipgram_remove300_10", False)
        predictions = fit_models(vectorizer, train_data_comments, test_data_comments, train_data_labels, test_data_labels)
        all_predictions = all_predictions + predictions.tolist()

        i = i + 1
        evaluation_metrics(test_data_labels, predictions, pretty_table, "iteration" + str(i))

    evaluation_metrics(comments["label"], all_predictions, pretty_table, "final")
    print(pretty_table)
    print_confusion_matrix(comments["label"], all_predictions)



def fit_models(vectorizer, train_data_comments, test_data_comments, train_data_labels, test_data_labels):
    pretty_table = PrettyTable(["Algorithm", "Accuracy", "Precision", "Recall", "F1_Score"])

    vectorized_train_comments = vectorizer.fit_transform(train_data_comments)
    vectorized_test_comments = vectorizer.transform(test_data_comments)

    model = SVC(C=1, kernel='linear')
    model = model.fit(vectorized_train_comments, train_data_labels)
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data_labels, predictions, pretty_table, "SVM")
    print_confusion_matrix(test_data_labels, predictions)

    print(pretty_table)
    print("")
    return predictions


def evaluation_metrics(true_sentiment, predicted_sentiment, pretty_table, algorithm):
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(['NEGATIVE', 'POSITIVE'])
    test_labels = label_binarizer.transform(true_sentiment)
    predict_labels = label_binarizer.transform(predicted_sentiment)
    accuracy_str = str(accuracy_score(true_sentiment, predicted_sentiment))
    precision_str = str(precision_score(test_labels, predict_labels))
    recall_str = str(recall_score(test_labels, predict_labels))
    f1_score_str = str(f1_score(test_labels, predict_labels))
    pretty_table.add_row([algorithm, accuracy_str, precision_str, recall_str, f1_score_str])
    return


def print_confusion_matrix(label, prediction):
    cf_matrix = confusion_matrix(label, prediction)
    print(cf_matrix.ravel())
    print(cf_matrix)


main()

# SVM W2V
# [[940  65]
#  [212 787]]
# [940,  65, 212, 787]  (tn, fp, fn, tp)

# RNN w2v
# [797 155 121 847]
# ('Accuracy: ', 0.85520833209156988)
# ('Precision: ', 0.84530938123752497)
# ('Recall: ', 0.875)
# ('F1 Score: ', 0.85989847715736034)

# svm w2v cross val
# [[2349  171]
#  [ 496 1994]]




# cbow 200 10 word count
# [908  97 211 788]
# [[908  97]
#  [211 788]]
# +-----------+-------------------+--------------------+--------------------+--------------------+
# | Algorithm |      Accuracy     |     Precision      |       Recall       |      F1_Score      |
# +-----------+-------------------+--------------------+--------------------+--------------------+
# |    SVM    | 0.846307385229541 | 0.8903954802259887 | 0.7887887887887888 | 0.8365180467091295 |
# +-----------+-------------------+--------------------+--------------------+--------------------+
# |    SVM    | 0.8398203592814372 | 0.8834841628959276 | 0.7817817817817818 | 0.8295273499734466 | tf-idf

# cbow 300 10 word count
# [899 106 202 797]
# [[899 106]
#  [202 797]]
# +-----------+-------------------+--------------------+--------------------+--------------------+
# | Algorithm |      Accuracy     |     Precision      |       Recall       |      F1_Score      |
# +-----------+-------------------+--------------------+--------------------+--------------------+
# |    SVM    | 0.846307385229541 | 0.8826135105204873 | 0.7977977977977978 | 0.8380651945320715 |
# +-----------+-------------------+--------------------+--------------------+--------------------+
# |    SVM    | 0.8383233532934131 | 0.8704720087815587 | 0.7937937937937938 | 0.8303664921465969 | tf-idf

# cbow 400 10 word count
# +-----------+-------------------+-------------------+--------------------+--------------------+
# | Algorithm |      Accuracy     |     Precision     |       Recall       |      F1_Score      |
# +-----------+-------------------+-------------------+--------------------+--------------------+
# |    SVM    | 0.841816367265469 | 0.884009009009009 | 0.7857857857857858 | 0.8320084790673027 |
# +-----------+-------------------+-------------------+--------------------+--------------------+
# |    SVM    | 0.8383233532934131 | 0.8770949720670391 | 0.7857857857857858 | 0.828933474128828 | tf-idf

# cbow 1000 10 word count
# [904 101 213 786]
# [[904 101]
#  [213 786]]
# +-----------+-------------------+--------------------+--------------------+--------------------+
# | Algorithm |      Accuracy     |     Precision      |       Recall       |      F1_Score      |
# +-----------+-------------------+--------------------+--------------------+--------------------+
# |    SVM    | 0.843313373253493 | 0.8861330326944757 | 0.7867867867867868 | 0.8335100742311771 |
# +-----------+-------------------+--------------------+--------------------+--------------------+
# |    SVM    | 0.8378243512974052 | 0.8711453744493393 | 0.7917917917917918 | 0.8295752490823284 | tf-idf