# coding: utf-8

import numpy as np
import pandas as pd
import datetime
from random import randint
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import tensorflow as tf


word2vec_model_name = "../../../corpus/analyzed/saved_models/word2vec_model_skipgram_300"

num_features = 300
max_sentence_length = 50

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 30000

labels = tf.placeholder(tf.int32, [batchSize, numClasses])
data = tf.placeholder(tf.float32, [batchSize, max_sentence_length, num_features])


def main():
    comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove.csv", ";")
    train_data, test_data = train_test_split(comments, test_size=0.4, random_state=0)
    train_data_vectors, train_data_labels = comments_to_vectors(train_data)
    test_data_vectors, test_data_labels = comments_to_vectors(test_data)

    batch_data, batch_label = get_batch(batchSize, train_data_vectors, train_data_labels)

    train_neural_network(train_data_vectors, train_data_vectors, train_data_labels, test_data_vectors, test_data_labels)

    print(train_data_vectors.shape)
    print(len(train_data_labels))
    print(test_data_vectors.shape)
    print(len(test_data_labels))
    print(batch_data.shape)

    return


def comments_to_vectors(data):
    model = word2vec.Word2Vec.load(word2vec_model_name)
    comment_vectors = []
    comment_labels = []
    for comment in data["comment"]:
        comment_vectors.append(get_sentence_vector(model, comment))
    for label in data["label"]:
        if label == "POSITIVE":
            comment_labels.append([1, 0])
        else:
            comment_labels.append([0, 1])
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


def neural_network_model(X):
    # tf.reset_default_graph()

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)

    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    return prediction


def train_neural_network(X, train_data, train_labels, test_data, test_labels):
    prediction  = neural_network_model(X)
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    for i in range(iterations):
        #Next Batch of reviews
        nextBatch, nextBatchLabels = get_batch(batchSize, train_data, train_labels);
        # nextBatch, nextBatchLabels = get_batch_order(batchSize, train_data, train_labels, i % 125);
        sess.run(optimizer, {data: nextBatch, labels: nextBatchLabels})

        #Write summary to Tensorboard
        if (i % 50 == 0):
            summary = sess.run(merged, {data: nextBatch, labels: nextBatchLabels})
            writer.add_summary(summary, i)

        #Save the network every 10,000 training iterations
        if (i % 10000 == 0 and i != 0):
            save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)
    writer.close()

    overall_accuracy = 0
    for i in range(80):
        nextBatch, nextBatchLabels = get_batch_order(batchSize, test_data, test_labels, i)
        overall_accuracy = overall_accuracy + (sess.run(accuracy, {data: nextBatch, labels: nextBatchLabels})) * 100
        print("Accuracy for this batch:", (sess.run(accuracy, {data: nextBatch, labels: nextBatchLabels})) * 100)
    print("Overall accuracy: ", overall_accuracy / 80)



main()