# coding: utf-8

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts
from gensim.test.utils import get_tmpfile
from prettytable import PrettyTable
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, utils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

trainData = pd.read_csv("../../../corpus/analyzed/train.csv", ";", quoting=3)
testData = pd.read_csv("../../../corpus/analyzed/test.csv", ";", quoting=3)
unlabeledData = pd.read_csv("../../../corpus/analyzed/comments_all.csv", header=0, delimiter=";", quoting=3)
print("Read %d labeled train reviews, %d labeled test reviews, %d un-labeled reviews\n" %
      (trainData["comment"].size, testData["comment"].size, unlabeledData["comment"].size))

num_features = 300
context = 10
cores = 8
doc2vec_model = "../../../corpus/analyzed/saved_models/doc2vec_model_skipgram_" \
                 + str(num_features) + "_" + str(context)


def main():
    generate_doc2vec_model()


def generate_doc2vec_model():
    comments_list = []
    for comment in unlabeledData["comment"]:
        comment_as_list = get_word_list(comment)
        comments_list.append(comment_as_list)

    comments = [TaggedDocument(str(comment), [i]) for i, comment in enumerate(comments_list)]

    # model = Doc2Vec(comments, vector_size=300, window=5, min_count=1, workers=8)
    # model.save(doc2vec_model)

    # model = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=cores)
    # model.build_vocab(comments)
    # for epoch in tqdm(range(10)):
    #     model.train(utils.shuffle(comments), total_examples=len(comments), epochs=1)
    #     model.alpha -= 0.002
    #     model.min_alpha = model.alpha

    model = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
    model.build_vocab(comments)
    for epoch in tqdm(range(10)):
        model.train(utils.shuffle(comments), total_examples=len(comments), epochs=1)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    # model = Doc2Vec.load(doc2vec_model)
    # # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    # vector = model.infer_vector(["නැහැ", 'ගෑණි', 'ඔබතුමන්ට'])
    # print(vector)

    train_vector = []
    for comment in trainData["comment"]:
        comment_word_list = get_word_list(comment)
        train_vector.append(model.infer_vector(comment_word_list))

    test_vector = []
    for comment in testData["comment"]:
        comment_word_list = get_word_list(comment)
        test_vector.append(model.infer_vector(comment_word_list))

    pretty_table = PrettyTable(["Algorithm", "Accuracy", "Precision", "Recall", "F1_Score"])
    # Logistic Regression model
    classi_model = LogisticRegression()
    classi_model = classi_model.fit(train_vector, trainData["label"])
    predictions = classi_model.predict(test_vector)
    evaluation_metrics(testData["label"], predictions, pretty_table, "Logistic Regression")

    model = DecisionTreeClassifier()
    model = model.fit(train_vector, trainData["label"])
    predictions = model.predict(test_vector)
    evaluation_metrics(testData["label"], predictions, pretty_table, "Decision Tree")

    model = GaussianNB()
    model = model.fit(train_vector, trainData["label"])
    predictions = model.predict(test_vector)
    evaluation_metrics(testData["label"], predictions, pretty_table, "Naive Bayes")

    model = SVC(C=1, kernel='linear')
    model = model.fit(train_vector, trainData["label"])
    predictions = model.predict(test_vector)
    evaluation_metrics(testData["label"], predictions, pretty_table, "SVM")

    print(pretty_table)

# split a comment into sentences of words
def get_word_list(comment):
    word_list = []
    raw_sentences = str(comment).split(".")
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 2:
            word_list.extend(raw_sentence.split())
    return word_list


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

main()
