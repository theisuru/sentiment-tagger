# coding: utf-8

import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import AblationVectorizer
import SentimentCommons as Commons
import W2VVectorizer


def main():
    comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove_all_punc.csv", ";")
    multi_class = False
    # comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_multi.csv", ";")
    # multi_class = True

    start_time = time.time()
    run_holdout(comments, multi_class)
    end_time = time.time()
    print("Time taken for the process: " + str(end_time - start_time))
    return


def run_holdout(comments, multi_class):
    w2v_model_name = "../../../corpus/analyzed/saved_models/word2vec_model_skipgram_remove300_10"
    fasttext_model_name = "../../../corpus/analyzed/saved_models/fasttext_model_skipgram_remove_300_10"

    train_data, test_data = train_test_split(comments, test_size=0.4, random_state=0)
    print("Processing dataset: " + str(train_data.columns.values))

    print("Extracting features with tfidf + FastText count for ablation study")
    vectorizer_tfidf = TfidfVectorizer(analyzer="word", tokenizer=lambda text: text.split())
    vectorizer_fasttext = W2VVectorizer.W2VVectorizer(fasttext_model_name, False)
    vectorizer = AblationVectorizer.AblationVectorizer([vectorizer_tfidf, vectorizer_fasttext])
    Commons.fit_models(vectorizer, train_data, test_data, multi_class)

    print("Extracting features with FastText + word2vec for ablation study")
    vectorizer_w2vec = W2VVectorizer.W2VVectorizer(w2v_model_name, False)
    vectorizer_fasttext = W2VVectorizer.W2VVectorizer(fasttext_model_name, False)
    vectorizer = AblationVectorizer.AblationVectorizer([vectorizer_w2vec, vectorizer_fasttext])
    Commons.fit_models(vectorizer, train_data, test_data, multi_class)

    print("Extracting features with tfidf + FastText + word2vec for ablation study")
    vectorizer_tfidf = TfidfVectorizer(analyzer="word", tokenizer=lambda text: text.split())
    vectorizer_w2vec = W2VVectorizer.W2VVectorizer(w2v_model_name, False)
    vectorizer_fasttext = W2VVectorizer.W2VVectorizer(fasttext_model_name, False)
    vectorizer = AblationVectorizer.AblationVectorizer([vectorizer_tfidf, vectorizer_w2vec, vectorizer_fasttext])
    Commons.fit_models(vectorizer, train_data, test_data, multi_class)

    return


main()
