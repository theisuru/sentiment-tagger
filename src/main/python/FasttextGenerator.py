# encoding: utf-8

import pandas as pd
import fasttext
from prettytable import PrettyTable

num_features = 300
context = 10
# output_model = "../../../corpus/analyzed/saved_models/fasttext_model_skipgram_remove" + str(num_features) + "_" + str(context)
output_model = "../../../corpus/analyzed/saved_models/xxx"
# input_file = '../../../corpus/analyzed/comments_para.csv'
input_file = '../../../corpus/analyzed/comments_all_everything_x.csv'


def main():
    generate_model()
    return


def generate_model():

    num_features = 300  # Word vector dimensionality
    # context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent word
    min_word_count = 1  # Minimum word count - if not occurred this much remove
    num_workers = 6  # Number of threads to run in parallel

    model = fasttext.skipgram(input_file=input_file,
                              output=output_model,
                              dim=num_features, ws=context, min_count=min_word_count, thread=num_workers)
    # model.init_sims(replace=True)  # If you don't plan to train the model any further
    # model.save(model)
    #
    # check_model_qulity(model, 'නැහැ')

    print(model['නැහැ'])
    print(model.words)
    return


def add_to_table(model, word, pretty_table):
    similar_words = []
    for s in model.most_similar(word):
        similar_words.append(s[0])

    pretty_table.add_column(word, similar_words)

def check_model_qulity(model, word):
    pretty_table = PrettyTable()
    add_to_table(model, 'නැහැ', pretty_table)
    add_to_table(model, 'ඔබට', pretty_table)
    add_to_table(model, 'වේවා', pretty_table)
    add_to_table(model, 'පිස්සු', pretty_table)
    add_to_table(model, 'හොඳයි', pretty_table)

    for s in model.most_similar(word):
        print(s[0].decode("utf-8"))


main()
