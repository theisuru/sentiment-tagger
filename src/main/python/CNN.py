import numpy as np
from random import randint
import tensorflow as tf

word2vec_model_name = "../../../corpus/analyzed/saved_models/word2vec_model_skipgram_300"

num_features = 300
max_sentence_length = 100

n_classes = 2
batch_size = 64
iterations = 10

# x = tf.placeholder('float', [None, 30000])
x = tf.placeholder(tf.float32, [64, 100, 300])
y = tf.placeholder(tf.float32, [64, 2])

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def main():
    train_data_vectors, train_data_labels, test_data_vectors, test_data_labels = load_vectors()
    print("Training CNN model...")
    train_neural_network(x, train_data_vectors, train_data_labels, test_data_vectors, test_data_labels)


def load_vectors():
    train_data_vectors = np.load('./vectors/train_data_vectors.npy')
    train_data_labels = np.load('./vectors/train_data_labels.npy')
    test_data_vectors = np.load('./vectors/test_data_vectors.npy')
    test_data_labels = np.load('./vectors/test_data_labels.npy')
    return train_data_vectors, train_data_labels, test_data_vectors, test_data_labels


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


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 8])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 8, 16])),
               'W_fc': tf.Variable(tf.random_normal([25 * 75 * 16, 256])),
               'out': tf.Variable(tf.random_normal([256, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([8])),
              'b_conv2': tf.Variable(tf.random_normal([16])),
              'b_fc': tf.Variable(tf.random_normal([256])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 100, 300, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 25 * 75 * 16])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x, train_data, train_labels, test_data, test_labels):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    print("Going to train the model........")
    hm_epochs = 50
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(iterations):
                epoch_x, epoch_y = get_batch(batch_size, train_data, train_labels)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        print("Going to test the trained module")
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        for i in range(30):
            test_data_batch, test_labels_batch = get_batch_order(batch_size, test_data, test_labels, i)
            print('Accuracy in batch ',i, ": ", accuracy.eval({x: test_data_batch, y: test_labels_batch}))


main()
