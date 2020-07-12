# coding: utf-8
import string

import pandas as pd
from gensim.models import word2vec

unlabeledData = pd.read_csv("../../../corpus/analyzed/comments_all.csv", header=0, delimiter=";", quoting=3)
print("Read %d un-labeled reviews\n" % (unlabeledData["comment"].size))

num_features = 300
context = 10
model = "skipgram"  # cbow, skipgram
remove_punctuations = False
word2vec_model = "../../../corpus/analyzed/saved_models/word2vec_" \
                 + model + "_" + str(remove_punctuations).lower() + "_" + str(num_features) + "_" + str(context)


def main():
    generate_word2vec_model()
    return


def generate_word2vec_model():
    if remove_punctuations:
        punc_remover = lambda x: str(x).translate(str.maketrans('', '', string.punctuation))
        unlabeledData['comment'] = unlabeledData['comment'].apply(punc_remover)

    comments = []
    for comment in unlabeledData["comment"]:
        comments += to_separate_sentences(comment)

    print("# of comments taken for building the model: " + str(len(comments)))

    # num_features = 1000  # Word vector dimensionality
    # context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    min_word_count = 1  # Minimum word count - if not occurred this much remove
    num_workers = 4  # Number of threads to run in parallel

    model = word2vec.Word2Vec(comments, workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling, sg=0)
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
