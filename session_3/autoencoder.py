import os
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf

from session_3.libs.datasets import MNIST

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


data = MNIST(split=[0.999, 0., 0.001])

input_layer_size = data.X.shape[1]
image_size = int(input_layer_size ** (1 / 2))


def get_summary_directory():
    summaries_path = Path.cwd() / 'summaries'
    summaries_path.mkdir(exist_ok=True)

    num_runs = 0
    for _ in summaries_path.iterdir():
        num_runs += 1

    summaries_path = summaries_path / str(num_runs + 1)
    summaries_path.mkdir()
    return summaries_path


def build_convolutional_network(inputs: tf.Tensor, layers: List[Tuple[int, int]] = ((16, 4), (16, 4), (16, 4), )):
    output = tf.reshape(inputs, [-1, image_size, image_size, 1])
    layers = list(layers)

    first_dim = tf.shape(inputs)[0]

    kernels = []
    shapes = []
    num_inputs = 1
    layer_num = 1

    for num_kernels, kernel_size in layers:
        with tf.variable_scope(f'encoder/layer{layer_num}'):
            shapes.append(output.get_shape().as_list())

            filters = tf.get_variable(
                name='filters',
                shape=[kernel_size, kernel_size, num_inputs, num_kernels],
                initializer=tf.random_normal_initializer(stddev=0.02)
            )

            kernels.append(filters)
            num_inputs = num_kernels

            output = tf.nn.relu(tf.nn.conv2d(
                output, filters, strides=[1, 2, 2, 1], padding='SAME'
            ))
        layer_num += 1

    for kernel, shape in zip(reversed(kernels), reversed(shapes)):
        with tf.variable_scope(f'decoder/layer{layer_num}'):
            if layer_num > 1:
                activation = tf.nn.relu
            else:
                activation = tf.nn.sigmoid

            output = activation(tf.nn.conv2d_transpose(
                output, kernel, tf.stack([first_dim, *shape[1:]]), strides=[1, 2, 2, 1], padding='SAME'
            ))

    return tf.reshape(output, [-1, input_layer_size])


def build_network(inputs: tf.Tensor,  layers: List[int] = (512, 256, 128, 64,)):
    layers = list(layers)
    num_inputs = input_layer_size

    weights = []
    output = inputs

    for layer_num, num_outputs in enumerate(layers):

        with tf.variable_scope(f'encoder/layer{layer_num}'):
            weight = tf.get_variable(
                name='weight',
                shape=[num_inputs, num_outputs],
                initializer=tf.random_normal_initializer(stddev=0.02)
            )
            weights.append(weight)

            output = tf.nn.relu(
                tf.matmul(output, weight)
            )

        num_inputs = num_outputs

    encoding_layer = output

    weights.reverse()

    layers.reverse()
    layers.pop(0)
    layers.append(input_layer_size)

    for layer_num, num_outputs in enumerate(layers):

        with tf.variable_scope(f'decoder/layer{layer_num}'):
            weight = tf.transpose(weights[layer_num])
            if layer_num < len(layers) - 1:
                output = tf.nn.relu(
                    tf.matmul(output, weight)
                )
            else:
                output = tf.math.sigmoid(
                    tf.matmul(output, weight)
                )
    with tf.name_scope('image_summaries'):
        tf.summary.histogram('inputs', inputs)
        tf.summary.histogram('outputs', output)
        tf.summary.histogram('encoding_layer', encoding_layer)
    return output


def train(inputs: tf.Tensor, outputs: tf.Tensor, learning_rate=0.001, batch_size=100, num_epochs=100):
    with tf.name_scope('cost_function'):
        cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(inputs, outputs), 1))
    print(tf.shape(cost))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    tf.summary.scalar('cost', cost)
    with tf.name_scope('image_summaries'):
        input_images = tf.reshape(inputs, [-1, image_size, image_size, 1], name='input_images')
        output_images = tf.reshape(outputs, [-1, image_size, image_size, 1], name='output_images')
        error = tf.abs(output_images - input_images)
        summary_image = tf.concat([input_images, output_images, error], 2)
        all_images = tf.reshape(summary_image, [1, -1, image_size * 3, 1])
        tf.summary.image('all_images', all_images, max_outputs=1)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        summary_writer = tf.summary.FileWriter(get_summary_directory(), session.graph)
        merge_summaries = tf.summary.merge_all()

        for epoch_num in range(num_epochs):
            print(f'running epoch {epoch_num+1} / {num_epochs}')
            for current_batch, _ in data.train.next_batch(batch_size):
                session.run(optimizer, feed_dict={inputs: current_batch})
            summary = session.run(merge_summaries, feed_dict={inputs: data.test.images})
            summary_writer.add_summary(summary, epoch_num)

        summary_writer.close()


if __name__ == '__main__':
    placeholder = tf.placeholder(tf.float32, [None, input_layer_size], name='inputs')
    network_output = build_convolutional_network(placeholder)
    train(placeholder, network_output)
