# coding: utf-8

import pandas as pd
from gensim.models import word2vec

trainData = pd.read_csv("../../../corpus/analyzed/train.csv", ";", quoting=3)
testData = pd.read_csv("../../../corpus/analyzed/test.csv", ";", quoting=3)
unlabeledData = pd.read_csv("../../../corpus/analyzed/comments_all_remove.csv", header=0, delimiter=";", quoting=3)
print("Read %d labeled train reviews, %d labeled test reviews, %d un-labeled reviews\n" %
      (trainData["comment"].size, testData["comment"].size, unlabeledData["comment"].size))

num_features = 300
context = 10
word2vec_model = "../../../corpus/analyzed/saved_models/word2vec_model_skipgram_remove" \
                 + str(num_features) + "_" + str(context)


def main():
    generate_word2vec_model()
    return


def generate_word2vec_model():
    comments = []
    for comment in unlabeledData["comment"]:
        comments += to_separate_sentences(comment)

    print("# of comments taken for building the model: " + str(len(comments)))

    # num_features = 1000  # Word vector dimensionality1
    # context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    min_word_count = 1  # Minimum word count - if not occurred this much remove
    num_workers = 4  # Number of threads to run in parallel

    model = word2vec.Word2Vec(comments, workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling, sg=1)
    model.init_sims(replace=True)  # If you don't plan to train the model any further
    model.save(word2vec_model)

    check_model_qulity(model, 'නැහැ')
    check_model_qulity(model, 'හොඳයි')
    check_model_qulity(model, 'ඔබට')
    return


# split a comment into sentences of words
def to_separate_sentences(comment):
    sentences = []
    raw_sentences = str(comment).split(".")
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 2:
            sentences.append(raw_sentence.split())
    return sentences


def check_model_qulity(model, word):
    for s in model.most_similar(word):
        print(s[0])
    # for s in model.wv.most_similar(positive=['ගෑණි', 'ඔබතුමන්ට'], negative=['මිනිහා']):
    #     print(s[0])


main()




Word presence & 0.84 & 0.83 & 0.85 & 0.82 & 0.83 & 0.84 & 0.84 & 0.82\ \
bigram presence & 0.72 & 0.68 & 0.66 & 0.62 & 0.90 & 0.93 & 0.76 & 0.74\ \
tf-idf & 0.85 &0.85 & 0.90 & 0.89 & 0.78 & 0.81 & 0.84 & 0.85 \ \
Word2Vec with mean embedding & 0.84 &0.84 & 0.88 & 0.88 & 0.79 & 0.79 & 0.83 & 0.84\ \
Word2Vec with tf-idf & 0.84 & 0.84 & 0.86 & 0.87 & 0.79 & 0.79 & 0.83 & 0.83\ \
fastText with mean embedding & 0.82 & 0.81 & 0.84 & 0.83 & 0.79 & 0.78 & 0.81 & 0.81\ \
fastText with tf-idf & 0.80 & 0.80 & 0.82 & 0.81 & 0.77 & 0.77 & 0.80 & 0.79\ \





Logistic Regression & 0.85 & 0.87 & 0.88 & 0.90 & 0.92 & 0.90 & 0.78 & 0.81 & 0.85 & 0.84 & 0.86 & 0.88\ \
Decision Tree & 0.75 & 0.75 & 0.72 & 0.76 & 0.75 & 0.72 & 0.74 & 0.72 & 0.72 & 0.75 & 0.75 & 0.72 \ \
Naive Bayes & 0.76 & 0.76 & 0.71 & 0.75 & 0.84 & 0.80 & 0.78 & 0.65 & 0.57 & 0.76 & 0.73 & 0.66 \ \
SVM & 0.85 & 0.87 & 0.88 & 0.89 & 0.92 & 0.89 & 0.81 & 0.80 & 0.86 & 0.84 & 0.86 & 0.87 \ \
Random Forest & 0.81 & 0.86 & 0.85 & 0.84 & 0.91 & 0.88 & 0.75 & 0.79 & 0.80 & 0.80 & 0.85 & 0.84 \ \





Algorithm & Accuracy & Precision & Recall & F1\_Score\\ \hline
Naive Bayes & 0.78 & 0.84 & 0.69 & 0.75\\ \hline
Decision Tree & 0.76 & 0.76 & 0.76 & 0.77\\ \hline
SVM (Word2Vec) & 0.87 & 0.92 & 0.80 & 0.86\\ \hline
SVM (fastText) & 0.88 & 0.89 & 0.86 & 0.87\\ \hline
RNN LSTM (Word2Vec) & 0.86 & 0.89 & 0.85 & 0.87\\ \hline
RNN LSTM (fastText) & 0.88 & 0.89 & 0.87 & 0.88\\ \hline
CNN SVM & 0.83 & 0.82 & 0.85 & 0.83\\ \hline
Logistic Regression (Word2Vec) & 0.87 & 0.92 & 0.81 & 0.86\\ \hline
Logistic Regression (fastText) & 0.88 & 0.90 & 0.85 & 0.88\\ \hline
Random Forest & 0.86 & 0.91 & 0.79 & 0.85\\ \hline
