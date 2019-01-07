import tensorflow as tf
import os
import shutil
from tensorflow.examples.tutorials.mnist import input_data

mnist_db = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batchsize = 20

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    #dropout ||||||| optional
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_session(x):
    print("begin training")

    prediction = convolutional_neural_network(x)  # one hot array
    print("PREDICTIONS: ",prediction)

    # calcola il distacco tra la soluzione tensorflow e reale
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # we could put an optimization rate but default (0.001) in enough in Admadoptimizer
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # numero totale di cicli (backpropagaation + feedforward)
    hm_epochs = 3

    # starting the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #starting cycles to train our model
        for epoch in range(hm_epochs):
            epoch_loss = 0
            #range is not fixed
            for _ in range(int(mnist_db.train.num_examples / batchsize)):
                epoch_x, epoch_y = mnist_db.train.next_batch(batchsize)
                # c stands for cost
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print("Epoch", epoch, "completed out of", hm_epochs, "loss:", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy: ", accuracy.eval({x: mnist_db.test.images[:1000], y: mnist_db.test.labels[:1000]}))




train_session(x)
print("training done")
