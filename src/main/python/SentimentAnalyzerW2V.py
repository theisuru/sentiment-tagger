# coding: utf-8

import SentimentCommons as Commons
import W2VVectorizer
import pandas as pd
import time
from prettytable import PrettyTable
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict


def main():
    start_time = time.time()
    run_holdout()
    end_time = time.time()
    print("Time taken for the process: " + str(end_time - start_time))
    return


def run_holdout():
    w2v_model_path = "../../../corpus/analyzed/saved_models/"
    comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove.csv", ";")
    train_data, test_data = train_test_split(comments, test_size=0.4, random_state=0)
    print("Processing dataset: " + str(train_data.columns.values))

    print("Extracting features with W2V count vectorizer")
    vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_path + "word2vec_model_from_unlabeled_comments_all_300", False)
    Commons.fit_models(vectorizer, train_data, test_data)

    print("Extracting features with W2V tfidf vectorizer")
    vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_path + "word2vec_model_from_unlabeled_comments_all_300", True)
    Commons.fit_models(vectorizer, train_data, test_data)

    print("Extracting features with W2V count vectorizer")
    vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_path + "word2vec_model_skipgram_300", False)
    Commons.fit_models(vectorizer, train_data, test_data)

    print("Extracting features with W2V tfidf vectorizer")
    vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_path + "word2vec_model_skipgram_300", True)
    Commons.fit_models(vectorizer, train_data, test_data)

    print("Extracting features with W2V count vectorizer")
    vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_path + "word2vec_model_skipgram_300_5", False)
    Commons.fit_models(vectorizer, train_data, test_data)

    print("Extracting features with W2V tfidf vectorizer")
    vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_path + "word2vec_model_skipgram_300_5", True)
    Commons.fit_models(vectorizer, train_data, test_data)

    return


# def run_cross_validation():
#     w2v_model_path = "../../../corpus/analyzed/saved_models/word2vec_model_skipgram_300"
#     comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove.csv", ";")
#
#     pretty_table = PrettyTable(["Algorithm", "Accuracy", "Precision", "Recall", "F1_Score"])
#     vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_path + "word2vec_model_skipgram_300", False)
#     model_svm = make_pipeline(vectorizer, SVC(C=1, kernel="linear"))
#
#     predictions = cross_val_predict(model_svm, comments["comment"], comments["label"], cv=3)
#     Commons.evaluation_metrics(comments["label"], predictions, pretty_table, "SVM")
#
#     comments["predictions"] = predictions
#     comments.to_csv("../../../corpus/analyzed/comments_predictions.csv", ";")
#
#     print("Cross validation results with tfidf")
#     print(pretty_table)

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
