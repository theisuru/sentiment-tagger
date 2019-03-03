# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns

from gensim.models import word2vec

trainData = pd.read_csv("../../../corpus/analyzed/train.csv", ";")
testData = pd.read_csv("../../../corpus/analyzed/test.csv", ";")
unlabeledTrainData = pd.read_csv("../../../corpus/analyzed/comments.csv", header=0, delimiter=";", quoting=3)
print("Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n" %
      (trainData["comment"].size, testData["comment"].size, unlabeledTrainData["comment"].size))


def to_separate_sentences(comment):
    sentences = []
    raw_sentences = comment.split(".")
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 2:
            sentences.append(raw_sentence.split())
    return sentences


comments = []
for comment in unlabeledTrainData["comment"]:
    comments += to_separate_sentences(comment)

print(len(comments))

num_features = 1000    # Word vector dimensionality
min_word_count = 1   # Minimum word count - if not occurred this much remove
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

model = word2vec.Word2Vec(comments, workers=num_workers, size=num_features, min_count = min_word_count,
                          window = context, sample = downsampling)
model.init_sims(replace=True)
model_name = "3000_article_comments"
model.save(model_name)
for s in model.most_similar('නැහැ'):
    print(s[0].decode("utf-8"))

# ----------------------- #
model = word2vec.Word2Vec.load("3000_article_comments")
print(model["නැහැ"])

