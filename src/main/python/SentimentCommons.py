import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def fit_models(vectorizer, train_data, test_data, multi_class=False):
    pretty_table = PrettyTable(["Algorithm", "Accuracy", "Precision", "Recall", "F1_Score"])

    vectorized_train_comments = vectorizer.fit_transform(train_data["comment"])
    vectorized_test_comments = vectorizer.transform(test_data["comment"])

    # Logistic Regression model
    model = LogisticRegression(solver='liblinear')
    # model = LogisticRegression(solver='lbfgs')
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Logistic Regression", multi_class)
    print_confusion_matrix(test_data["label"], predictions)

    # Decision Tree  model
    model = DecisionTreeClassifier()
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Decision Tree", multi_class)

    if isinstance(vectorized_train_comments, csr_matrix):
        vectorized_train_comments_dense = vectorized_train_comments.toarray()
        vectorized_test_comments_dense = vectorized_test_comments.toarray()
    else:
        vectorized_train_comments_dense = vectorized_train_comments
        vectorized_test_comments_dense = vectorized_test_comments

    # Naive Bayes  model
    model = GaussianNB()
    model = model.fit(vectorized_train_comments_dense, train_data["label"])
    predictions = model.predict(vectorized_test_comments_dense)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Naive Bayes", multi_class)

    # Support Vector Machine  model
    # model = SVC(probability=True, kernel='rbf')
    model = SVC(C=1, kernel='linear')
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "SVM", multi_class)
    # print_confusion_matrix(test_data["label"], predictions)

    # Random forest model
    model = RandomForestClassifier(n_estimators=100)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Random Forest", multi_class)

    print(pretty_table)
    print("")
    return


def fit_models_ablation(vectorizer_1, vectorizer_2 , train_data, test_data, multi_class=False):
    pretty_table = PrettyTable(["Algorithm", "Accuracy", "Precision", "Recall", "F1_Score"])

    vectorized_train_comments_1 = vectorizer_1.fit_transform(train_data["comment"]).toarray()
    vectorized_train_comments_2 = vectorizer_2.fit_transform(train_data["comment"])
    vectorized_train_comments = np.concatenate((vectorized_train_comments_1, vectorized_train_comments_2), axis=1)

    vectorized_test_comments_1 = vectorizer_1.transform(test_data["comment"]).toarray()
    vectorized_test_comments_2 = vectorizer_2.transform(test_data["comment"])
    vectorized_test_comments = np.concatenate((vectorized_test_comments_1, vectorized_test_comments_2), axis=1)

    # Logistic Regression model
    model = LogisticRegression(solver='liblinear')
    # model = LogisticRegression(solver='lbfgs')
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Logistic Regression", multi_class)
    print_confusion_matrix(test_data["label"], predictions)

    # Decision Tree  model
    model = DecisionTreeClassifier()
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Decision Tree", multi_class)

    if isinstance(vectorized_train_comments, csr_matrix):
        vectorized_train_comments_dense = vectorized_train_comments.toarray()
        vectorized_test_comments_dense = vectorized_test_comments.toarray()
    else:
        vectorized_train_comments_dense = vectorized_train_comments
        vectorized_test_comments_dense = vectorized_test_comments

    # Naive Bayes  model
    model = GaussianNB()
    model = model.fit(vectorized_train_comments_dense, train_data["label"])
    predictions = model.predict(vectorized_test_comments_dense)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Naive Bayes", multi_class)

    # Support Vector Machine  model
    # model = SVC(probability=True, kernel='rbf')
    model = SVC(C=1, kernel='linear')
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "SVM", multi_class)

    # Random forest model
    model = RandomForestClassifier(n_estimators=100)
    model = model.fit(vectorized_train_comments, train_data["label"])
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data["label"], predictions, pretty_table, "Random Forest", multi_class)

    print(pretty_table)
    print("")
    return


def evaluation_metrics(true_sentiment, predicted_sentiment, pretty_table, algorithm, multi_class):
    label_binarizer = preprocessing.LabelBinarizer()
    if multi_class:
        labels = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        label_binarizer.fit(labels)
        average = None
        # average = 'weighted'
    else:
        labels = ['POSITIVE', 'NEGATIVE']
        label_binarizer.fit(labels)
        average = 'binary'

    test_labels = label_binarizer.transform(true_sentiment)
    predict_labels = label_binarizer.transform(predicted_sentiment)
    accuracy_str = str(accuracy_score(true_sentiment, predicted_sentiment))
    precision_str = str(precision_score(test_labels, predict_labels, labels=labels, average=average))
    recall_str = str(recall_score(test_labels, predict_labels, labels=labels, average=average))
    f1_score_str = str(f1_score(test_labels, predict_labels, labels=labels, average=average))
    # precision_str = str(precision_score(test_labels, predict_labels, average=None))
    # recall_str = str(recall_score(test_labels, predict_labels, average=None))
    # f1_score_str = str(f1_score(test_labels, predict_labels, average=None))
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
    cf_matrix = confusion_matrix(label, prediction, ["POSITIVE", "NEGATIVE", "NEUTRAL"])
    print(cf_matrix)
