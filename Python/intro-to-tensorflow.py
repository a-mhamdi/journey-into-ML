#= conda activate introML =# `tf__version__ = 1.14.0`  

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

"""
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""

# tf.reset_default_graph()

x = tf.placeholder(shape=[2, 1], dtype=tf.float32)
W = tf.get_variable(name='W', shape=[2, 2], dtype=tf.float32, initializer=tf.random_normal_initializer)
b = tf.constant([[0],[1]], dtype=tf.float32)

y = tf.add(tf.matmul(W, x), b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(W))
    print(sess.run(y, feed_dict={x: [[1], [.5]]}))
