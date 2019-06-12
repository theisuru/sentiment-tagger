# encoding: utf-8

import pandas as pd
import fasttext

num_features = 300
context = 5
model = "../../../corpus/analyzed/saved_models/fasttext_model_skipgram_" \
        + str(num_features) + "_" + str(context)


def main():
    generate_model()
    return


def generate_model():

    num_features = 300  # Word vector dimensionality
    # context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent word
    min_word_count = 1  # Minimum word count - if not occurred this much remove
    num_workers = 6  # Number of threads to run in parallel

    model = fasttext.skipgram(input_file='../../../corpus/analyzed/comments_para.csv',
                              output='../../../corpus/analyzed/saved_models/fasttext_model_skipgram_300',
                              dim=num_features, ws=context, min_count=min_word_count, thread=num_workers)
    # model.init_sims(replace=True)  # If you don't plan to train the model any further
    # model.save(model)
    #
    # check_model_qulity(model, 'නැහැ')

    print(model['නැහැ'])
    print(model.words)
    return


def check_model_qulity(model, word):
    for s in model.most_similar(word):
        print(s[0].decode("utf-8"))


main()
