from typing import Tuple
from pprint import pprint
from pathlib import Path

from skimage.data import astronaut
from skimage.transform import resize
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
xs = np.linspace([-3], [3], n_observations)
ys = np.reshape(np.sin(xs) + np.random.uniform(-0.5, 0.5, [n_observations, 1]), -1)

im_size = 64
img = resize(astronaut(), [im_size, im_size])

img_x = []
img_y = []
for row_i in range(img.shape[0]):
    for col_i in range(img.shape[1]):
        # And store the inputs
        img_x.append([row_i, col_i])
        # And outputs that the network needs to learn to predict
        img_y.append(img[row_i, col_i])

img_x = np.array(img_x)
img_y = np.array(img_y)

img_x = (img_x - np.mean(img_x)) / np.std(img_x)


def distance(p1, p2):
    with tf.name_scope('distance'):
        return tf.abs(tf.subtract(p1, p2, name='subtraction'), name='absolute_value')


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

        summary_writer = tf.summary.FileWriter(summaries_path, sess.graph)
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
                batch_dict = {X: feed_dict[X][batch_inds], Y: feed_dict[Y][batch_inds]}
                sess.run(optimize, feed_dict=batch_dict)
                iter_num += 1
                training_cost, summaries = sess.run([cost, merge], feed_dict=feed_dict)

                cost_log.append(training_cost)
                summary_writer.add_summary(summaries, iter_num)

                if iter_num % 10 == 0:
                    ys_predicted = Y_predicted.eval(feed_dict=feed_dict, session=sess)

                    ax1.plot(xs, ys_predicted, 'k', alpha=iter_num / total_runs)
                    plt.draw()

            if np.abs(prev_cost - training_cost) < 1E-6:
                break

            prev_cost = training_cost
        summary_writer.close()
    ax2.plot(cost_log)
    fig.show()
    plt.draw()


def init_placeholders():
    X = tf.placeholder(tf.float32, name='X', shape=[None, None])
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


def neural_network(layers=(1, 10,)):
    X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')

    output = X

    for layer_num in range(1, len(layers)):
        output = linear(
            output,
            layers[layer_num-1],
            layers[layer_num],
            activation=tf.nn.relu if (layer_num+1) < len(layers) else None,
            scope=f'layer_{layer_num}'
        )

    Y_predicted = output

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
        outputs = tf.matmul(inputs, weight, name='layer_multiplication') + bias

        if activation is not None:
            outputs = activation(outputs)
        return outputs


def image_train(X, Y, Y_predicted, n_runs=500, batch_size=50, learning_rate=0.001):

    summaries_path = Path.cwd() / 'summaries'
    if not summaries_path.exists():
        summaries_path.mkdir(parents=True)

    num_runs = 0
    for _ in summaries_path.iterdir():
        num_runs += 1

    summaries_path = summaries_path / str(num_runs + 1)
    summaries_path.mkdir()

    with tf.name_scope('cost'):
        cost = tf.reduce_mean(
            tf.reduce_sum(distance(Y_predicted, Y), 1))

    tf.summary.scalar('cost', cost)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    feed_dict = {X: img_x, Y: img_y}

    tf.summary.image('image', tf.reshape(Y_predicted, [1, *img.shape], 'flattened_image'))
    tf.summary.image('expected_image', tf.reshape(Y, [1, *img.shape], 'flattened_expected_image'))
    session_config = tf.ConfigProto(
        log_device_placement=True
    )
    with tf.Session(config=session_config) as sess:

        summary_writer = tf.summary.FileWriter(summaries_path, sess.graph)
        merge = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())

        for run_num in range(n_runs):
            indices = np.random.permutation(range(len(img_x)))
            n_batches = len(indices) // batch_size
            for batch_start in range(0, n_batches, batch_size):
                batch_inds = indices[batch_start:batch_start+batch_size]
                batch_dict = {X: feed_dict[X][batch_inds], Y: feed_dict[Y][batch_inds]}
                sess.run(optimizer, feed_dict=batch_dict)

            training_cost, reports = sess.run([cost, merge], feed_dict=feed_dict)
            summary_writer.add_summary(reports, run_num)
            print(f'{run_num:0>3}: {training_cost:7.3f}')


if __name__ == '__main__':
    # plt.imshow(img)
    # plt.show()

    n_inputs = 2
    n_hidden_layers = 6
    neurons_per_hidden = 64
    n_outputs = 3

    layers = [n_inputs]
    layers.extend([neurons_per_hidden for _ in range(n_hidden_layers)])
    layers.append(n_outputs)

    network = neural_network(layers)
    graph = tf.get_default_graph()
    op: tf.Operation
    pprint([op.name for op in graph.get_operations()])
    image_train(*network)
