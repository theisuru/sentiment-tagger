# coding: utf-8

import time

import pandas as pd
from prettytable import PrettyTable
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

import SentimentCommons as Commons
import W2VVectorizer
import D2VVectorizer


def main():
    start_time = time.time()
    # run_train_test_split()
    run_holdout()
    # run_cross_validation()
    end_time = time.time()
    print("Time taken for the process: " + str(end_time - start_time))
    return


def run_train_test_split():
    train_data = pd.read_csv("../../../corpus/analyzed/train.csv", ";")
    test_data = pd.read_csv("../../../corpus/analyzed/test.csv", ";")
    print("Processing dataset: " + str(train_data.columns.values))

    print("Extracting features with count vectorizer")
    vectorizer = CountVectorizer(analyzer="word", tokenizer=lambda text: text.split())
    Commons.fit_models(vectorizer, train_data, test_data)

    print("Extracting features with tfidf vectorizer")
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=lambda text: text.split())
    Commons.fit_models(vectorizer, train_data, test_data)

    print("Extracting features with W2V count vectorizer")
    vectorizer = W2VVectorizer.W2VVectorizer(
        "../../../corpus/analyzed/saved_models/word2vec_model_skipgram_300_10", False)
    Commons.fit_models(vectorizer, train_data, test_data)

    print("Extracting features with D2V vectorizer")
    vectorizer = D2VVectorizer.D2VVectorizer(
        "../../../corpus/analyzed/saved_models/doc2vec_model_skipgram_200_10")
    Commons.fit_models(vectorizer, train_data, test_data)


def run_holdout():
    w2v_model_name = "../../../corpus/analyzed/saved_models/word2vec_model_skipgram_300_5"
    fasttext_model_name = "../../../corpus/analyzed/saved_models/fasttext_model_skipgram_remove_300_10"
    # w2v_model_name = "../../../corpus/analyzed/saved_models/word2vec_model_from_unlabeled_comments_all_300"
    comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove_all_punc.csv", ";")
    train_data, test_data = train_test_split(comments, test_size=0.4, random_state=0)
    print("Processing dataset: " + str(train_data.columns.values))

    print("Extracting features with count vectorizer")
    vectorizer = CountVectorizer(analyzer="word", tokenizer=lambda text: text.split())
    Commons.fit_models(vectorizer, train_data, test_data)

    print("Extracting features with 2-gram count vectorizer")
    vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer="word", tokenizer=lambda text: text.split())
    Commons.fit_models(vectorizer, train_data, test_data)
    #
    print("Extracting features with tfidf vectorizer")
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=lambda text: text.split())
    Commons.fit_models(vectorizer, train_data, test_data)

    print("Extracting features with W2V count vectorizer")
    vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_name, False)
    Commons.fit_models(vectorizer, train_data, test_data)

    print("Extracting features with W2V tfidf vectorizer")
    vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_name, True)
    Commons.fit_models(vectorizer, train_data, test_data)

    print("Extracting features with FastText count vectorizer")
    vectorizer = W2VVectorizer.W2VVectorizer(fasttext_model_name, False)
    Commons.fit_models(vectorizer, train_data, test_data)

    print("Extracting features with FastText tfidf vectorizer")
    vectorizer = W2VVectorizer.W2VVectorizer(fasttext_model_name, True)
    Commons.fit_models(vectorizer, train_data, test_data)

    return


def run_cross_validation():
    comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove.csv", ";")
    pretty_table = PrettyTable(["Algorithm", "Accuracy", "Precision", "Recall", "F1_Score"])
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=lambda text: text.split())
    model_svm = make_pipeline(vectorizer, SVC(C=1, kernel="linear"))
    model_lr = make_pipeline(vectorizer, LogisticRegression())

    # scores = cross_val_score(model, comments["comment"], comments["label"], cv=3, scoring="f1_macro")
    # print(scores)

    # predictions = cross_val_predict(model_lr, comments["comment"], comments["label"], cv=3)
    # Commons.evaluation_metrics(comments["label"], predictions, pretty_table, "Logistic Regression")
    predictions = cross_val_predict(model_svm, comments["comment"], comments["label"], cv=3)
    Commons.evaluation_metrics(comments["label"], predictions, pretty_table, "SVM")

    comments["predictions"] = predictions
    comments.to_csv("../../../corpus/analyzed/comments_predictions.csv", ";")

    print("Cross validation results with tfidf")
    print(pretty_table)

    return


main()
