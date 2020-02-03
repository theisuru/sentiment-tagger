import numpy as np
import pandas as pd
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import string

word2vec_model_name = "../../../corpus/analyzed/saved_models/word2vec_model_skipgram_300"
comment_file = "../../../corpus/analyzed/comments_all.csv"
comment_remove_punc_file = "../../../corpus/analyzed/comments_all_remove.csv"

num_features = 300
max_sentence_length = 100

batchSize = 25
lstmUnits = 64
numClasses = 2
iterations = 30000

def main():
    # convert_to_vectors()
    remove_punctuations()


def remove_punctuations():
    comments = pd.read_csv(comment_file, delimiter=';')
    punc_remover = lambda x : str(x).translate(str.maketrans('', '', string.punctuation))
    comments['comment'] = comments['comment'].apply(punc_remover)
    comments.to_csv(comment_remove_punc_file, sep=';', index=False)


def convert_to_vectors():
    comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove.csv", ";")
    train_data, test_data = train_test_split(comments, test_size=0.4, random_state=0)
    train_data_vectors, train_data_labels = comments_to_vectors(train_data)
    test_data_vectors, test_data_labels = comments_to_vectors(test_data)

    np.savetxt('./vectors/train_data_vectors.txt', train_data_vectors)
    np.savetxt('./vectors/train_data_labels.txt', train_data_labels)
    np.savetxt('./vectors/test_data_vectors.txt', test_data_vectors)
    np.savetxt('./vectors/test_data_labels.txt', test_data_labels)


def comments_to_vectors(data):
    model = word2vec.Word2Vec.load(word2vec_model_name)
    comment_vectors = []
    comment_labels = []
    for comment in data["comment"]:
        comment_vectors.append(get_sentence_vector(model, comment))
    for label in data["label"]:
        if label == "POSITIVE":
            comment_labels.append([0, 1])
        else:
            comment_labels.append([1, 0])
    return np.array(comment_vectors), comment_labels


def get_sentence_vector(model, sentence):
    sentence_vector = np.zeros([max_sentence_length, num_features])
    counter = 0
    index2word_set = set(model.wv.index2word)
    for word in sentence.split():
        if word in index2word_set:
            sentence_vector[counter] = model[word]
            counter += 1
            if counter == max_sentence_length:
                break
        else:
            print("word not in word2vec model: " + word)
    return sentence_vector


main()