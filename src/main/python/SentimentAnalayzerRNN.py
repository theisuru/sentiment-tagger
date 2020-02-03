# coding: utf-8

import numpy as np
import pandas as pd
import datetime
from random import randint
from gensim.models import word2vec
from gensim.models.fasttext import FastText
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import string


# based on https://github.com/adeshpande3/LSTM-Sentiment-Analysis/blob/master/Oriole%20LSTM.ipynb

word2vec_model_name = "../../../corpus/analyzed/saved_models/word2vec_model_skipgram_remove300_10"
# word2vec_model_name = "../../../corpus/analyzed/saved_models/fasttext_model_skipgram_remove_300_10"
# word2vec_model_name = "../../../corpus/analyzed/saved_models/wiki.si.bin"

num_features = 300
max_sentence_length = 50

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 30000

labels = tf.placeholder(tf.int32, [batchSize, numClasses])
data = tf.placeholder(tf.float32, [batchSize, max_sentence_length, num_features])


def main():
    convert_to_vectors()
    train_data_vectors, train_data_labels, test_data_vectors, test_data_labels = load_vectors()

    print("Running tesnsorflow simulation.....")
    loss, accuracy, prediction_values, optimizer = neural_network_model()
    train_neural_network(loss, accuracy, optimizer, train_data_vectors, train_data_labels)
    accuracy, precision, recall, f1 = measure_neural_network(accuracy, prediction_values, test_data_vectors, test_data_labels)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)


def convert_to_vectors():
    comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove_all_punc.csv", ";")
    punc_remover = lambda x : str(x).translate(str.maketrans('', '', string.punctuation))
    comments['comment'] = comments['comment'].apply(punc_remover)
    train_data, test_data = train_test_split(comments, test_size=0.4, random_state=0)
    train_data_vectors, train_data_labels = comments_to_vectors(train_data)
    test_data_vectors, test_data_labels = comments_to_vectors(test_data)

    np.save('./vectors/train_data_vectors.npy', train_data_vectors)
    np.save('./vectors/train_data_labels.npy', train_data_labels)
    np.save('./vectors/test_data_vectors.npy', test_data_vectors)
    np.save('./vectors/test_data_labels.npy', test_data_labels)


def load_vectors():
    train_data_vectors = np.load('./vectors/train_data_vectors.npy')
    train_data_labels = np.load('./vectors/train_data_labels.npy')
    test_data_vectors = np.load('./vectors/test_data_vectors.npy')
    test_data_labels = np.load('./vectors/test_data_labels.npy')
    return train_data_vectors, train_data_labels, test_data_vectors, test_data_labels


def comments_to_vectors(data):
    model = word2vec.Word2Vec.load(word2vec_model_name)  #loading word2vec model, this is the correct old one
    # model = FastText.load_fasttext_format("../../../corpus/analyzed/saved_models/wiki.si.bin")  #loading word2vec model
    # model = FastText.load_fasttext_format("../../../corpus/analyzed/saved_models/fasttext_model_skipgram_300.bin")  #loading word2vec model
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
            if (counter == max_sentence_length):
                break
        else:
            print("word not in word2vec model: " + word)
    return sentence_vector


def get_batch(size, data, label):
    batch_data = np.empty((size, max_sentence_length, num_features), dtype=float)
    batch_label = []
    for i in range(size):
        random_int = randint(0, len(data) - 1)
        batch_data[i] = data[random_int]
        batch_label.append(label[random_int])
    return batch_data, batch_label


def get_batch_order(size, data, label, batch_no):
    batch_data = data[batch_no * size : (batch_no + 1) * size]
    batch_label = label[batch_no * size : (batch_no + 1) * size]
    return batch_data, batch_label


def neural_network_model():
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)

    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    prediction_values = tf.argmax(prediction, 1)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return loss, accuracy, prediction_values, optimizer


def train_neural_network(loss, accuracy, optimizer, train_data, train_labels):
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    for i in range(iterations):
        # Next Batch of reviews
        next_batch, next_batch_labels = get_batch(batchSize, train_data, train_labels)
        sess.run(optimizer, {data: next_batch, labels: next_batch_labels})

        # Write summary to Tensorboard
        if (i % 50 == 0):
            summary = sess.run(merged, {data: next_batch, labels: next_batch_labels})
            writer.add_summary(summary, i)

        # Save the network every 10,000 training iterations
        if (i % 9999 == 0 and i != 0):
            save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)
    writer.close()


def measure_neural_network(accuracy, prediction_values, test_data, test_labels):
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('models'))

    overall_accuracy = 0
    all_predictions = []
    test_iterations = 80
    for i in range(test_iterations):
        next_batch, next_batch_labels = get_batch_order(batchSize, test_data, test_labels, i)
        accuracy_this_batch = (sess.run(accuracy, {data: next_batch, labels: next_batch_labels})) * 100
        predictions_this_batch = sess.run(prediction_values, {data: next_batch, labels: next_batch_labels})
        overall_accuracy = overall_accuracy + accuracy_this_batch
        all_predictions = all_predictions + predictions_this_batch.tolist()
        print("Accuracy for this batch:", accuracy_this_batch)

    true_labels = tf.argmax(test_labels, 1).eval()
    precision = precision_score(true_labels.tolist()[0:batchSize * test_iterations], all_predictions)
    f1 = f1_score(true_labels.tolist()[0:batchSize * test_iterations], all_predictions)
    recall = recall_score(true_labels.tolist()[0:batchSize * test_iterations], all_predictions)
    overall_accuracy = overall_accuracy / (test_iterations * 100)
    print(confusion_matrix(true_labels.tolist()[0:batchSize * test_iterations], all_predictions).ravel())

    return overall_accuracy, precision, recall, f1


main()


# 0.891712707182
# 0.853146853146853
#
# fn = tp(1-0.891712707182)/0.891712707182
# fp = tp(1-0.853146853146853)/0.853146853146853
#
# fn = tp(0.12143742255306716)
# fp = tp(0.1721311475409838)
#
#
# fn = 885*(1-0.891712707182)/0.891712707182
# fp = 885*(1-0.853146853146853)/0.853146853146853

#
# fasttext
# ('Accuracy: ', 0.8619791641831398)
# ('Precision: ', 0.8772874058127018)
# ('Recall: ', 0.8419421487603306)
# ('F1 Score: ', 0.8592514496573538)

# ('Accuracy: ', 0.8661458320915699)
# ('Precision: ', 0.8967813540510544)
# ('Recall: ', 0.8347107438016529)
# ('F1 Score: ', 0.864633493846977)

# skipgram 300_10
# [852 100 160 808]
# Accuracy:  0.8651041679084301
# Precision:  0.8898678414096917
# Recall:  0.8347107438016529
# F1 Score:  0.861407249466951


# gensim.fastext 300_10 homemade
# [865  87 159 809]
# Accuracy:  0.8697916679084301
# Precision:  0.9029017857142857
# Recall:  0.8357438016528925
# F1 Score:  0.8680257510729613

# [855  97 143 825]
# Accuracy:  0.8750000044703483
# Precision:  0.8947939262472885
# Recall:  0.8522727272727273
# F1 Score:  0.873015873015873

# fasttext pretrained
# [821 131 195 773]
# Accuracy:  0.8333333313465119
# Precision:  0.8550884955752213
# Recall:  0.7985537190082644
# F1 Score:  0.8258547008547009

# [803 149 200 768]
# Accuracy:  0.8192708320915699
# Precision:  0.8375136314067612
# Recall:  0.7933884297520661
# F1 Score:  0.8148541114058356

# fasttext homemade
# [872  80 171 797]
# Accuracy:  0.8703125007450581
# Precision:  0.9087799315849487
# Recall:  0.8233471074380165
# F1 Score:  0.8639566395663957

# [861  91 148 820]
# Accuracy:  0.8770833320915699
# Precision:  0.9001097694840834
# Recall:  0.8471074380165289
# F1 Score:  0.8728046833422033


# fastext gensim homemade no punc
# Accuracy:  0.8817708335816861
# Precision:  0.8927813163481954
# Recall:  0.868801652892562
# F1 Score:  0.8806282722513089