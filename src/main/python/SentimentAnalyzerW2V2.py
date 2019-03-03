# coding: utf-8

import numpy as np
import pandas as pd
import math
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score

word2vec_model = "../../../corpus/analyzed/saved_models/word2vec_model_from_unlabeled_comments_all_1000"
trainData = pd.read_csv("../../../corpus/analyzed/train.csv", ";", quoting=3)
testData = pd.read_csv("../../../corpus/analyzed/test.csv", ";", quoting=3)
unlabeledData = pd.read_csv("../../../corpus/analyzed/comments_all.csv", header=0, delimiter=";", quoting=3)
print("Read %d labeled train reviews, %d labeled test reviews, %d un-labeled reviews\n" %
      (trainData["comment"].size, testData["comment"].size, unlabeledData["comment"].size))

no_of_train_samples = trainData.size

def main():
    # generate_word2vec_model()
    train_labels = trainData["label"]
    test_labels = testData["label"]

    train_data_vecs, test_data_vecs = get_train_test_data_vecs(False)
    print("train data size = %d, test data size = %d\n" % (train_data_vecs.size, test_data_vecs.size))
    clssify_using_random_forest(train_data_vecs, test_data_vecs, train_labels, test_labels)
    clssify_using_svm(train_data_vecs, test_data_vecs, train_labels, test_labels)
    clssify_using_logistic_regression(train_data_vecs, test_data_vecs, train_labels, test_labels)
    clssify_using_naive_bayes(train_data_vecs, test_data_vecs, train_labels, test_labels)

    train_data_vecs, test_data_vecs = get_train_test_data_vecs(True)
    print("train data size = %d, test data size = %d\n" % (train_data_vecs.size, test_data_vecs.size))
    clssify_using_random_forest(train_data_vecs, test_data_vecs, train_labels, test_labels)
    clssify_using_svm(train_data_vecs, test_data_vecs, train_labels, test_labels)
    clssify_using_logistic_regression(train_data_vecs, test_data_vecs, train_labels, test_labels)
    clssify_using_naive_bayes(train_data_vecs, test_data_vecs, train_labels, test_labels)
    return


def calculate_idf(train_comments):
    print("calculation idf scores")
    index2word_set = set()
    word2doc_frequency = {}
    for comment in train_comments:
        comment_word_set = set()
        for word in comment.split():
            if word in index2word_set:
                if word not in comment_word_set:
                    word2doc_frequency[word] = word2doc_frequency.get(word) + 1
            else:
                index2word_set.add(word)
                word2doc_frequency[word] = 1
                comment_word_set.add(word)
    return word2doc_frequency


# split a comment into sentences of words
def to_separate_sentences(comment):
    sentences = []
    raw_sentences = str(comment).split(".")
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 2:
            sentences.append(raw_sentence.split())
    return sentences

# to word list
def to_word_list(comment):
    return comment.split()


# make a feature vector from a single comment
def make_feature_vec(words, model, num_features):
    feature_vec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec, model[word])

    # we have some one word comments that is not included in original model, todo expand original model or remove them
    if nwords != 0:
        feature_vec = np.divide(feature_vec, nwords)

    return feature_vec


# make a tfidf feature vector from a single comment
def make_feature_vec_tfidf(words, model, num_features, word2doc_frequency):
    feature_vec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set and word in word2doc_frequency:
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec, model[word] * math.log10(no_of_train_samples / (word2doc_frequency.get(word))))

    # we have some one word comments that is not included in original model, todo expand original model or remove them
    if nwords != 0:
        feature_vec = np.divide(feature_vec, nwords)

    return feature_vec


# get a list of feature vectors for all comments
def get_avg_feature_vecs(reviews, model, num_features):
    counter = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        if counter % 1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))

        review_feature_vecs[int(counter)] = make_feature_vec(review, model, num_features)
        counter = counter + 1.
    return review_feature_vecs


# get a list of feature vectors for all comments
def get_avg_feature_vecs_tfidf(reviews, model, num_features, word2doc_frequency):
    counter = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        if counter % 1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))

        review_feature_vecs[int(counter)] = make_feature_vec_tfidf(review, model, num_features, word2doc_frequency)
        counter = counter + 1.
    return review_feature_vecs


def get_train_test_data_vecs(tfidf):
    num_features = 1000
    model = word2vec.Word2Vec.load("word2vec_model_from_unlabeled_comments_all")

    clean_train_comments = []
    clean_test_comments = []

    for comment in trainData["comment"]:
        clean_train_comments.append(comment.split())
    for comment in testData["comment"]:
        clean_test_comments.append(comment.split())

    if tfidf:
        word2doc_frequency = calculate_idf(trainData["comment"])
        train_data_vecs = get_avg_feature_vecs_tfidf(clean_train_comments, model, num_features, word2doc_frequency)
        test_data_vecs = get_avg_feature_vecs_tfidf(clean_test_comments, model, num_features, word2doc_frequency)
    else:
        train_data_vecs = get_avg_feature_vecs(clean_train_comments, model, num_features)
        test_data_vecs = get_avg_feature_vecs(clean_test_comments, model, num_features)

    return train_data_vecs, test_data_vecs


def clssify_using_random_forest(train_data_vec, test_data_vec, train_labels, test_labels):
    forest = RandomForestClassifier(n_estimators=100)
    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(train_data_vec, train_labels)
    result = forest.predict(test_data_vec)

    confusion_matrix = pd.crosstab(test_labels, result, rownames=["Actual"], colnames=["Predicted"])
    print(confusion_matrix)

    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(['NEGATIVE', 'POSITIVE'])
    test_sentiment = label_binarizer.transform(test_labels)
    predict_sentiment = label_binarizer.transform(result)
    accuracy_str = str(accuracy_score(test_sentiment, predict_sentiment))
    precision_str = str(precision_score(test_sentiment, predict_sentiment))
    f1_score_str = str(f1_score(test_sentiment, predict_sentiment))
    print("Accuracy = %s \nPrecision = %s \nF1score = %s \n" % (accuracy_str, precision_str, f1_score_str))
    return


def clssify_using_svm(train_data_vec, test_data_vec, train_labels, test_labels):
    svm = SVC(C=1, kernel='linear')
    print("Fitting a SVM to labeled training data...")
    svm = svm.fit(train_data_vec, train_labels)
    result = svm.predict(test_data_vec)

    confusion_matrix = pd.crosstab(test_labels, result, rownames=["Actual"], colnames=["Predicted"])
    print(confusion_matrix)

    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(['NEGATIVE', 'POSITIVE'])
    test_sentiment = label_binarizer.transform(test_labels)
    predict_sentiment = label_binarizer.transform(result)
    accuracy_str = str(accuracy_score(test_sentiment, predict_sentiment))
    precision_str = str(precision_score(test_sentiment, predict_sentiment))
    f1_score_str = str(f1_score(test_sentiment, predict_sentiment))
    print("Accuracy = %s \nPrecision = %s \nF1score = %s \n" % (accuracy_str, precision_str, f1_score_str))
    return


def clssify_using_logistic_regression(train_data_vec, test_data_vec, train_labels, test_labels):
    logistic_regression = LogisticRegression()
    print("Fitting a logistic regression to labeled training data...")
    logistic_regression = logistic_regression.fit(train_data_vec, train_labels)
    result = logistic_regression.predict(test_data_vec)

    confusion_matrix = pd.crosstab(test_labels, result, rownames=["Actual"], colnames=["Predicted"])
    print(confusion_matrix)

    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(['NEGATIVE', 'POSITIVE'])
    test_sentiment = label_binarizer.transform(test_labels)
    predict_sentiment = label_binarizer.transform(result)
    accuracy_str = str(accuracy_score(test_sentiment, predict_sentiment))
    precision_str = str(precision_score(test_sentiment, predict_sentiment))
    f1_score_str = str(f1_score(test_sentiment, predict_sentiment))
    print("Accuracy = %s \nPrecision = %s \nF1score = %s \n" % (accuracy_str, precision_str, f1_score_str))
    return


def clssify_using_naive_bayes(train_data_vec, test_data_vec, train_labels, test_labels):
    naive_bayes = GaussianNB()
    print("Fitting a naive bayes to labeled training data...")
    naive_bayes = naive_bayes.fit(train_data_vec, train_labels)
    result = naive_bayes.predict(test_data_vec)

    confusion_matrix = pd.crosstab(test_labels, result, rownames=["Actual"], colnames=["Predicted"])
    print(confusion_matrix)

    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(['NEGATIVE', 'POSITIVE'])
    test_sentiment = label_binarizer.transform(test_labels)
    predict_sentiment = label_binarizer.transform(result)
    accuracy_str = str(accuracy_score(test_sentiment, predict_sentiment))
    precision_str = str(precision_score(test_sentiment, predict_sentiment))
    f1_score_str = str(f1_score(test_sentiment, predict_sentiment))
    print("Accuracy = %s \nPrecision = %s \nF1score = %s \n" % (accuracy_str, precision_str, f1_score_str))
    return


def generate_word2vec_model():
    comments = []
    for comment in unlabeledData["comment"]:
        comments += to_separate_sentences(comment)

    print(len(comments))

    num_features = 1000  # Word vector dimensionality
    min_word_count = 1  # Minimum word count - if not occurred this much remove
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    model = word2vec.Word2Vec(comments, workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)
    model.init_sims(replace=True)  # If you don't plan to train the model any further
    model_name = "word2vec_model_from_unlabeled_comments_all"
    model.save(model_name)
    for s in model.most_similar('නැහැ'):
        print(s[0].decode("utf-8"))
    return


main()
