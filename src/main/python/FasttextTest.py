# encoding: utf-8

import pandas as pd
from gensim.models import word2vec
from gensim.models.fasttext import FastText
from prettytable import PrettyTable
import fasttext


num_features = 300
context = 10
fasttext_model = "../../../corpus/analyzed/saved_models/fasttext_model_skipgram_" \
                 + str(num_features) + "_" + str(context)


def main():
    pretty_table = PrettyTable()

    # model = load_gensim_fastext()
    # add_to_table(model, 'නැහැ', 'gensim fasttext', pretty_table)  #නැහැ, හොඳයි, ඔබට
    model = load_homemade_fasttext()
    add_to_table(model, 'නැහැ', 'homemade fasttext', pretty_table)
    add_to_table(model, 'හොඳයි', 'homemade fasttext', pretty_table)
    add_to_table(model, 'ඔබට', 'homemade fasttext', pretty_table)
    # model = load_pretrained_fasttext()
    # add_to_table(model, 'නැහැ', 'pretrained fasttext', pretty_table)

    print(pretty_table)
    return


def load_gensim_fastext():
    return word2vec.Word2Vec.load(fasttext_model)


def load_homemade_fasttext():
    return FastText.load_fasttext_format("../../../corpus/analyzed/saved_models/fasttext_model_skipgram_300.bin")


def load_pretrained_fasttext():
    return FastText.load_fasttext_format("../../../corpus/analyzed/saved_models/wiki.si.bin")


def add_to_table(model, word, model_name, pretty_table):
    similar_words = []
    for s in model.most_similar(word):
        similar_words.append(s[0])

    pretty_table.add_column(model_name, similar_words)

main()

