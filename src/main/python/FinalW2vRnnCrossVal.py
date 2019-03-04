# coding: utf-8

import SentimentCommons as Commons
import W2VVectorizer
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from prettytable import PrettyTable
from sklearn.svm import SVC
import tensorflow as tf
import datetime
from random import randint
from gensim.models import word2vec


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
    start_time = time.time()
    run_cross_val()
    end_time = time.time()
    print("Time taken for the process: " + str(end_time - start_time))
    return


def convert_to_vectors():
    comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove.csv", ";")
    data_vectors, data_labels = comments_to_vectors(comments)

    np.save('./vectors/data_vectors.npy', data_vectors)
    np.save('./vectors/data_labels.npy', data_labels)


def load_vectors():
    data_vectors = np.load('./vectors/data_vectors.npy')
    data_labels = np.load('./vectors/data_labels.npy')
    return data_vectors, data_labels


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
        #Next Batch of reviews
        next_batch, next_batch_labels = get_batch(batchSize, train_data, train_labels)
        sess.run(optimizer, {data: next_batch, labels: next_batch_labels})

        #Write summary to Tensorboard
        if (i % 50 == 0):
            summary = sess.run(merged, {data: next_batch, labels: next_batch_labels})
            writer.add_summary(summary, i)

        #Save the network every 10,000 training iterations
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
    test_iterations = 20
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

    all_test = true_labels.tolist()[0:batchSize * test_iterations]
    return overall_accuracy, precision, recall, f1, all_predictions, all_test


def run_cross_val():
    all_predictions = []
    all_used_test_labels = []
    w2v_model_path = "../../../corpus/analyzed/saved_models/"
    comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove.csv", ";")
    pretty_table = PrettyTable(["Algorithm", "Accuracy", "Precision", "Recall", "F1_Score"])

    # convert_to_vectors()
    data_vectors, data_labels = load_vectors()

    print("Running tesnsorflow simulation.....")


    i = 1
    kf = KFold(n_splits=10)
    kf.get_n_splits(comments)
    for train_index, test_index in kf.split(data_vectors):
        train_data_comments, test_data_comments = data_vectors[train_index], data_vectors[test_index]
        train_data_labels, test_data_labels = data_labels[train_index], data_labels[test_index]


        tf.reset_default_graph()

        global labels
        global data
        labels = tf.placeholder(tf.int32, [batchSize, numClasses])
        data = tf.placeholder(tf.float32, [batchSize, max_sentence_length, num_features])

        loss, accuracy, prediction_values, optimizer = neural_network_model()
        train_neural_network(loss, accuracy, optimizer, train_data_comments, train_data_labels)
        accuracy, precision, recall, f1, predictions, used_test_labels = measure_neural_network(accuracy, prediction_values, test_data_comments, test_data_labels)

        all_predictions = all_predictions + predictions
        all_used_test_labels = all_used_test_labels + used_test_labels

        i = i + 1
        evaluation_metrics(used_test_labels, predictions, pretty_table, "iteration" + str(i))

    evaluation_metrics(all_used_test_labels, all_predictions, pretty_table, "final")
    print(pretty_table)
    print_confusion_matrix(all_used_test_labels, all_predictions)



def fit_models(vectorizer, train_data_comments, test_data_comments, train_data_labels, test_data_labels):
    pretty_table = PrettyTable(["Algorithm", "Accuracy", "Precision", "Recall", "F1_Score"])

    vectorized_train_comments = vectorizer.fit_transform(train_data_comments)
    vectorized_test_comments = vectorizer.transform(test_data_comments)

    model = SVC(C=1, kernel='linear')
    model = model.fit(vectorized_train_comments, train_data_labels)
    predictions = model.predict(vectorized_test_comments)
    evaluation_metrics(test_data_labels, predictions, pretty_table, "SVM")
    print_confusion_matrix(test_data_labels, predictions)

    print(pretty_table)
    print("")
    return predictions


def evaluation_metrics(true_sentiment, predicted_sentiment, pretty_table, algorithm):
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(['NEGATIVE', 'POSITIVE'])
    test_labels = label_binarizer.transform(true_sentiment)
    predict_labels = label_binarizer.transform(predicted_sentiment)
    accuracy_str = str(accuracy_score(true_sentiment, predicted_sentiment))
    precision_str = str(precision_score(true_sentiment, predicted_sentiment))
    recall_str = str(recall_score(true_sentiment, predicted_sentiment))
    f1_score_str = str(f1_score(true_sentiment, predicted_sentiment))
    pretty_table.add_row([algorithm, accuracy_str, precision_str, recall_str, f1_score_str])
    return


def print_confusion_matrix(label, prediction):
    cf_matrix = confusion_matrix(label, prediction)
    print(cf_matrix.ravel())
    print(cf_matrix)


main()

# SVM W2V
# [[940  65]
#  [212 787]]
# [940,  65, 212, 787]  (tn, fp, fn, tp)

# RNN w2v
# [797 155 121 847]
# ('Accuracy: ', 0.85520833209156988)
# ('Precision: ', 0.84530938123752497)
# ('Recall: ', 0.875)
# ('F1 Score: ', 0.85989847715736034)

# svm w2v cross val
# [[2349  171]
#  [ 496 1994]]

# [2105  295  360 2040]

# 1.0 * (797+847) / (797+155+121+847)
# 1.0 * (2349+1994) / (2349+1994+496+171)
