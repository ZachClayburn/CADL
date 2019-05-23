from typing import Tuple
from pprint import pprint
from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

plt.style.use('ggplot')

n_observations = 1000
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)


def distance(p1, p2):
    return tf.abs(p1 - p2)


def train(X, Y, Y_predicted, n_runs=100, batch_size=200, learning_rate=0.02):
    summaries_path = Path.cwd() / 'summaries'
    if not summaries_path.exists():
        summaries_path.mkdir(parents=True)

    num_runs = 0
    for _ in summaries_path.iterdir():
        num_runs += 1

    summaries_path = summaries_path / str(num_runs + 1)
    summaries_path.mkdir()

    fig: plt.Figure
    fig, ax = plt.subplots(1, 2)
    ax1: plt.Axes = ax[0]
    ax2: plt.Axes = ax[1]
    cost_log = []

    ax1.scatter(xs, ys, alpha=0.5, marker='+')
    with tf.Session() as sess:
        cost = tf.reduce_mean(distance(Y_predicted, Y), name='cost_fun')
        tf.summary.scalar('cost', cost)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimize = optimizer.minimize(cost)

        summary_writter = tf.summary.FileWriter(summaries_path, sess.graph)
        merge = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        prev_cost = 0
        iter_num = 0
        n_batches = len(xs) // batch_size
        total_runs = n_batches * n_runs
        for _ in range(n_runs):
            feed_dict = {X: xs, Y: ys}

            inds = np.random.permutation(range(len(xs)))
            for batch_num in range(n_batches):
                batch_inds = inds[batch_num * batch_size: (batch_num + 1) * batch_size]
                sess.run(optimize, feed_dict={X: xs[batch_inds], Y: ys[batch_inds]})
                iter_num += 1
                training_cost, summaries = sess.run([cost, merge], feed_dict=feed_dict)

                cost_log.append(training_cost)
                summary_writter.add_summary(summaries, iter_num)

                if iter_num % 10 == 0:
                    ys_predicted = Y_predicted.eval(feed_dict=feed_dict, session=sess)

                    ax1.plot(xs, ys_predicted, 'k', alpha=iter_num / total_runs)
                    plt.draw()

            if np.abs(prev_cost - training_cost) < 1E-6:
                break

            prev_cost = training_cost
        summary_writter.close()
    ax2.plot(cost_log)
    fig.show()
    plt.draw()


def init_placeholders():
    X = tf.placeholder(tf.float32, name='X', shape=[None])
    Y = tf.placeholder(tf.float32, name='Y', shape=[None])
    return X, Y


def simple_line_network(size=1) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    X, Y = init_placeholders()

    weight = tf.Variable(tf.random_normal([1, size], stddev=0.1, dtype=tf.float32), name='weight')
    # weight = tf.Variable(tf.constant(1, shape=[1], dtype=tf.float32), name='weight')
    bias = tf.Variable(tf.constant(0, shape=[size], dtype=tf.float32), name='bias')
    # bias = tf.Variable(tf.random_normal([1], stddev=0.1, dtype=tf.float32), name='bias')

    Y_predicted = tf.reduce_sum(tf.matmul(tf.expand_dims(X, 1), weight) + bias, 1)
    print(Y_predicted.shape)
    return X, Y, Y_predicted


def polynomial_network(order=1):
    X, Y = init_placeholders()

    Y_predicted = tf.Variable(tf.random_normal([1]), name='bias')
    for power in range(1, order + 1):
        weight = tf.Variable(
            tf.random_normal([1], stddev=0.1), name=f'weight_{power}'
        )
        Y_predicted = tf.add(tf.multiply(tf.pow(X, power), weight), Y_predicted)
    return X, Y, Y_predicted


def neural_network(layers=(10,)):
    X, Y = init_placeholders()

    output = tf.expand_dims(X, 1)
    prev_n_neurons = 1
    layer_num = 1
    for n_neurons in layers:
        output = linear(output, prev_n_neurons, n_neurons, activation=tf.nn.tanh, scope=f'layer_{layer_num}')
        prev_n_neurons = n_neurons
        layer_num += 1

    Y_predicted = tf.reduce_sum(output, 1)

    return X, Y, Y_predicted


def linear(inputs, n_inputs, n_outputs, activation=None, scope=None):
    with tf.variable_scope(scope or 'linear'):
        weight = tf.get_variable(
            name='weight',
            shape=[n_inputs, n_outputs],
            initializer=tf.random_normal_initializer(stddev=1.)
        )
        tf.summary.histogram('weight', weight)
        bias = tf.get_variable(
            name='bias',
            shape=[n_outputs],
            initializer=tf.constant_initializer()
        )
        tf.summary.histogram('bias', bias)
        outputs = tf.matmul(inputs, weight) + bias

        if activation is not None:
            outputs = activation(outputs)
        return outputs


if __name__ == '__main__':
    network = neural_network((10, 10))
    graph = tf.get_default_graph()
    op: tf.Operation
    pprint([op.name for op in graph.get_operations()])
    train(*network, learning_rate=0.01, n_runs=300)
