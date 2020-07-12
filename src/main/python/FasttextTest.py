# encoding: utf-8

import pandas as pd
from gensim.models import word2vec
from gensim.models.fasttext import FastText
from prettytable import PrettyTable
import fasttext


num_features = 300
context = 10
fasttext_model = "../../../corpus/analyzed/saved_models/fasttext_model_skipgram_remove_" + str(num_features) + "_" + str(context)


def main():
    pretty_table = PrettyTable()

    # model = load_gensim_fastext()
    model = load_homemade_fasttext()
    # model = load_pretrained_fasttext()
    # model = load_w2v()

    add_to_table(model, 'නැහැ', pretty_table)
    add_to_table(model, 'ඔබට', pretty_table)
    add_to_table(model, 'වේවා', pretty_table)
    add_to_table(model, 'පිස්සු', pretty_table)
    add_to_table(model, 'හොඳයි',  pretty_table)

    print(pretty_table)
    return


def load_w2v():
    return word2vec.Word2Vec.load("../../../corpus/analyzed/saved_models/word2vec_model_skipgram_remove300_10")


def load_gensim_fastext():
    return word2vec.Word2Vec.load("../../../corpus/analyzed/saved_models/x")
    # return word2vec.Word2Vec.load(fasttext_model)


def load_homemade_fasttext():
    # return FastText.load_fasttext_format("../../../corpus/analyzed/saved_models/fasttext_model_skipgram_300.bin")
    # return FastText.load_fasttext_format("../../../corpus/analyzed/saved_models/fasttext_model_skipgram_remove300_5.bin")
    return FastText.load_fasttext_format("../../../corpus/analyzed/saved_models/xxx.bin")


def load_pretrained_fasttext():
    return FastText.load_fasttext_format("../../../corpus/analyzed/saved_models/wiki.si.bin")


def add_to_table(model, word, pretty_table):
    similar_words = []
    for s in model.most_similar(word):
        similar_words.append(s[0])

    pretty_table.add_column(word, similar_words)

main()
