import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from math import pi as PI
from skimage import data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    with tf.Session().as_default():
        x = tf.expand_dims(tf.linspace(-3.0, 3.0, 100), 1)

        mean = 0.0
        std_dev = 1.0

        z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                                (2.0 * tf.pow(std_dev, 2.0)))) *
             (1.0 / (std_dev * tf.sqrt(2.0 * PI))))

        gaussian = tf.matmul(z, z, transpose_b=True)

        plt.imshow(gaussian.eval())
        plt.title('Gaussian Kernel')
        plt.colorbar()
        plt.show()

        sin = tf.tile(tf.sin(x), [*x.shape[::-1]])
        gabor = tf.multiply(sin, gaussian)

        plt.imshow(gabor.eval())
        plt.title('Gabor Kernel')
        plt.show()

        img = data.camera().astype(np.float32)

        plt.imshow(img, cmap='gray')
        plt.title('Before convolution')
        plt.show()

        kernel = tf.placeholder(tf.float32, [None, None], name='kernel')
        kernel_shape = tf.shape(kernel)
        gaussian_4d = tf.reshape(gaussian, [kernel_shape[0], kernel_shape[1], 1, 1])

        img_placeholder = tf.placeholder(tf.float32, [None, None], name='image')
        img_shape = tf.shape(img_placeholder)
        img_4d = tf.reshape(img_placeholder, [1, img_shape[0], img_shape[1], 1])

        convolved: tf.Tensor = tf.nn.conv2d(img_4d, gaussian_4d, strides=[1, 1, 1, 1], padding='SAME')

        feed = {
            'image:0': img,
            'kernel:0': gaussian.eval()
        }
        result = convolved.eval(feed)

        plt.imshow(np.squeeze(result), cmap='gray')
        plt.title('gaussian convolution')
        plt.show()

        feed['kernel:0'] = gabor.eval()
        result = convolved.eval(feed)

        plt.imshow(np.squeeze(result), cmap='gray')
        plt.title('gabor convolution')
        plt.show()


