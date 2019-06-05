# coding: utf-8

import time

import matplotlib.pyplot as plt
import pandas as pd
from prettytable import PrettyTable
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import SentimentCommons as Commons
import W2VVectorizer


def main():
    start_time = time.time()
    run_holdout()
    end_time = time.time()
    print("Time taken for the process: " + str(end_time - start_time))
    return


def run_holdout():
    w2v_model_name = "../../../corpus/analyzed/saved_models/word2vec_model_skipgram_300_5"
    # w2v_model_name = "../../../corpus/analyzed/saved_models/fasttext_model_skipgram"
    comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove_all_punc.csv", ";")
    train_data, test_data = train_test_split(comments, test_size=0.4, random_state=0)
    print("Processing dataset: " + str(train_data.columns.values))

    # print("Extracting features with tfidf vectorizer")
    # vectorizer = TfidfVectorizer(analyzer="word", tokenizer=lambda text: text.split())
    # fit_models_random_forest(vectorizer, train_data, test_data)
    # fit_models_svm(vectorizer, train_data, test_data)
    # fit_models_logistic_regression(vectorizer, train_data, test_data)

    print("Extracting features with W2V count vectorizer")
    vectorizer = W2VVectorizer.W2VVectorizer(w2v_model_name, False)
    fit_models_random_forest(vectorizer, train_data, test_data)
    fit_models_svm(vectorizer, train_data, test_data)
    fit_models_logistic_regression(vectorizer, train_data, test_data)

    return


def fit_models_random_forest(vectorizer, train_data, test_data):
    pretty_table = PrettyTable(["Algorithm", "Accuracy", "Precision", "Recall", "F1_Score"])

    vectorized_train_comments = vectorizer.fit_transform(train_data["comment"])
    vectorized_test_comments = vectorizer.transform(test_data["comment"])

    # Random forest model
    model = RandomForestClassifier(n_estimators=50)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Random Forest50")

    # Random forest model
    model = RandomForestClassifier(n_estimators=100)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Random Forest100")

    # Random forest model
    model = RandomForestClassifier(n_estimators=150)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Random Forest150")

    # Random forest model
    model = RandomForestClassifier(n_estimators=200)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Random Forest200")

    # Random forest model
    model = RandomForestClassifier(n_estimators=100, criterion='entropy')
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Random Forest100_entropy")

    # Random forest model
    model = RandomForestClassifier(n_estimators=150, criterion='entropy')
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Random Forest150_entropy")

    # Random forest model
    model = RandomForestClassifier(n_estimators=200, criterion='entropy')
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Random Forest200_entropy")

    # Random forest model
    model = RandomForestClassifier(n_estimators=150, max_features=None)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Random Forest150_none")

    # Random forest model
    model = RandomForestClassifier(n_estimators=200, max_features=None)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Random Forest200_none")

    # Random forest model
    model = RandomForestClassifier(n_estimators=150, criterion='entropy', max_features=None)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Random Forest150_entropy_none")

    # Random forest model
    model = RandomForestClassifier(n_estimators=200, criterion='entropy', max_features=None)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Random Forest200_entropy_none")

    print(pretty_table)
    print("")
    return


def fit_models_svm(vectorizer, train_data, test_data):
    pretty_table = PrettyTable(["Algorithm", "Accuracy", "Precision", "Recall", "F1_Score"])

    vectorized_train_comments = vectorizer.fit_transform(train_data["comment"])
    vectorized_test_comments = vectorizer.transform(test_data["comment"])

    # Support Vector Machine  model
    model = SVC(C=1, kernel='linear')
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "SVM_linear")
    print_confusion_matrix(test_data["label"], predictions)

    # Support Vector Machine  model
    model = SVC(kernel='rbf')
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "SVM_rbf")
    print_confusion_matrix(test_data["label"], predictions)

    # Support Vector Machine  model
    model = SVC(kernel='poly')
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "SVM_poly")
    print_confusion_matrix(test_data["label"], predictions)

    # Support Vector Machine  model
    model = SVC(kernel='sigmoid')
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "SVM_sigmoid")
    print_confusion_matrix(test_data["label"], predictions)

    # Support Vector Machine  model
    # model = SVC(kernel='precomputed')#sparse precomputed kernels are not supported
    # model = model.fit(vectorized_train_comments, train_data["label"])
    # predictions = model.predict(vectorized_test_comments)
    # evaluation_metrics(test_data["label"], predictions, pretty_table, "SVM_precomputed")
    # print_confusion_matrix(test_data["label"], predictions)

    # Support Vector Machine  model
    model = SVC(kernel='linear', probability=True)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "SVM_probability_output")
    print_confusion_matrix(test_data["label"], predictions)

    # Support Vector Machine  model
    model = SVC(kernel='linear', C=0.5)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "SVM_0.5")
    print_confusion_matrix(test_data["label"], predictions)

    # Support Vector Machine  model
    model = SVC(kernel='linear', C=1.5)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "SVM_1.5")
    print_confusion_matrix(test_data["label"], predictions)

    # Support Vector Machine  model
    model = SVC(kernel='linear', C=2)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "SVM_2")
    print_confusion_matrix(test_data["label"], predictions)

    # Support Vector Machine  model
    model = SVC(kernel='linear', C=3)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "SVM_3")
    print_confusion_matrix(test_data["label"], predictions)

    # Support Vector Machine  model
    model = SVC(kernel='linear', C=4)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "SVM_4")
    print_confusion_matrix(test_data["label"], predictions)

    print(pretty_table)
    print("")
    return


def fit_models_logistic_regression(vectorizer, train_data, test_data):
    pretty_table = PrettyTable(["Algorithm", "Accuracy", "Precision", "Recall", "F1_Score"])

    vectorized_train_comments = vectorizer.fit_transform(train_data["comment"])
    vectorized_test_comments = vectorizer.transform(test_data["comment"])

    # Logistic Regression model
    model = LogisticRegression()
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Logistic Regression")

    # Logistic Regression model
    model = LogisticRegression(penalty='l1')
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Logistic Regression")

    # Logistic Regression model
    model = LogisticRegression(tol=1e-5, C=1)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Logistic Regression")

    # Logistic Regression model
    model = LogisticRegression(tol=1e-4, C=0.5)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Logistic Regression")

    # Logistic Regression model
    model = LogisticRegression(tol=1e-4, C=1)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Logistic Regression")

    # Logistic Regression model
    model = LogisticRegression(tol=1e-4, C=2)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Logistic Regression")

    # Logistic Regression model
    model = LogisticRegression(tol=1e-4, C=3)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Logistic Regression")

    # Logistic Regression model
    model = LogisticRegression(tol=1e-4, C=4)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Logistic Regression")

    # Logistic Regression model
    model = LogisticRegression(penalty='l1', C=4)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Logistic Regression")

    print(pretty_table)
    print("")
    return


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


def compare_algorithms(true_sentiment, predicted_sentiment, vectorizer, model):
    confusion_matrix = pd.crosstab(true_sentiment, predicted_sentiment, rownames=["Actual"], colnames=["Predicted"])
    print(confusion_matrix)
    confusion_matrix.plot.bar(stacked=True)
    plt.legend(title='mark')
    plt.show()

    test_words = vectorizer.get_feature_names()
    # print(testWords[len(testWords) - 1])
    model_coeffs = model.coef_.tolist()[0]
    coeffdf = pd.DataFrame({'Word': test_words, 'Coefficient': model_coeffs})
    coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
    print(coeffdf.tail(10))
    print(coeffdf.head(10))
    return


def print_confusion_matrix(label, prediction):
    cf_matrix = confusion_matrix(label, prediction)
    print(cf_matrix)


main()
