# coding: utf-8

import numpy as np
import pandas as pd
import datetime
from random import randint
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score

# https://github.com/adeshpande3/LSTM-Sentiment-Analysis/blob/master/Oriole%20LSTM.ipynb

word2vec_model_name = "../../../corpus/analyzed/saved_models/word2vec_model_skipgram_300"

num_features = 300
max_sentence_length = 100
filter_sizes = [3, 4, 5]
num_filters = 64

batchSize = 25
numClasses = 2
iterations = 30000

dropout_keep_prob = 0.5

labels = tf.placeholder(tf.int32, [batchSize, numClasses])
data = tf.placeholder(tf.float32, [batchSize, max_sentence_length, num_features])


def main():
    # convert_to_vectors()
    train_data_vectors, train_data_labels, test_data_vectors, test_data_labels = load_vectors()

    batch_vectors, batch_labels = get_batch(batchSize, train_data_vectors, train_data_labels)
    batch_vectors_extended = tf.expand_dims(batch_vectors, -1)
    print(batch_vectors_extended.shape)

    # print("Running tesnsorflow simulation.....")
    # loss, accuracy, prediction_values, optimizer = neural_network_model()
    # train_neural_network(loss, accuracy, optimizer, train_data_vectors, train_data_labels)
    # accuracy, precision, f1 = test_neural_network(accuracy, prediction_values, test_data_vectors, test_data_labels)
    # print("Accuracy: ", accuracy)
    # print("Precision: ", precision)
    # print("F1 Score: ", f1)


def convert_to_vectors():
    comments = pd.read_csv("../../../corpus/analyzed/comments_tagged_remove.csv", ";")
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


def neural_network_model(train_data_vectors, train_labels):

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, num_features, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                train_data_vectors,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Max-pooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, max_sentence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

                # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    with tf.name_scope("output"):
        W = tf.Variable(tf.truncated_normal([num_filters_total, numClasses], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[numClasses]), name="b")
        scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
        predictions = tf.argmax(scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
    with tf.name_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(scores, train_labels)
        loss = tf.reduce_mean(losses)

        # Calculate Accuracy
    with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(predictions, tf.argmax(train_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    # session

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.scalar_summary("loss", cnn.loss)
    acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

    # Train Summaries
    train_summary_op = tf.merge_summary([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

    # Dev summaries
    dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

    # Checkpointing
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    # Tensorflow assumes this directory already exists so we need to create it
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())

    sess.run(tf.initialize_all_variables())



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


def test_neural_network(accuracy, prediction_values, test_data, test_labels):
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
    overall_accuracy = overall_accuracy / (test_iterations * 100)

    return overall_accuracy, precision, f1


main()


