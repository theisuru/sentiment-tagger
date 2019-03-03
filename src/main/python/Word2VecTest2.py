# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns

from gensim.models import word2vec

# ----------------------- #
model = word2vec.Word2Vec.load("../../../corpus/analyzed/saved_models/word2vec_model_skipgram_300")
# for s in model.most_similar('නැහැ'):
#     print(s[0].decode("utf-8"))
# for s in model.most_similar('හොඳයි'):
#     print(s[0].decode("utf-8"))
# for s in model.most_similar('පිස්සු'):
#     print(s[0].decode("utf-8"))
# for s in model.most_similar('වේවා'):
#     print(s[0].decode("utf-8"))
# for s in model.most_similar('ඔබට'):
#     print(s[0].decode("utf-8"))


for s in model.most_similar('මිනිහා'):
    print(s[0].decode("utf-8"))

for s in model.wv.most_similar(positive=['ගෑණි', 'ඔබතුමන්ට'], negative=['මිනිහා']):
    print(s[0].decode("utf-8"))

for s in model.wv.most_similar_cosmul(positive=['ගෑණි', 'ඔබතුමන්ට'], negative=['මිනිහා']):
    print(s[0].decode("utf-8"))

