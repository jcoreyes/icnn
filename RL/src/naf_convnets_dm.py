import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

padding = 'SAME'
num_channels = 3
width, height = FLAGS.width, FLAGS.height
flat_dim = 8*8*32
strides1 = [1, 2, 2, 1]
strides2 = [1, 2, 2, 1]

def theta(dimIn, dimOut, l1, l2, scope):
    # with tf.variable_scope(scope):
    #     normal_init = tf.truncated_normal_initializer(mean=0.0, stddev=FLAGS.initstd)
    #     return [tf.get_variable(name='w1', shape=[dimIn, l1], initializer=normal_init),
    #             tf.get_variable(name='b1', shape=[l1], initializer=tf.constant_initializer(0.0)),
    #             tf.get_variable(name='w2', shape=[l1, l2], initializer=normal_init),
    #             tf.get_variable(name='b2', shape=[l2], initializer=tf.constant_initializer(0.0)),
    #             tf.get_variable(name='w3', shape=[l2, dimOut], initializer=normal_init),
    #             tf.get_variable(name='b3', shape=[dimOut], initializer=tf.constant_initializer(0.0))]


    conv1filter = FLAGS.conv1filter
    conv1numfilters = FLAGS.conv1numfilters
    conv2filter = FLAGS.conv2filter
    conv2numfilters = FLAGS.conv2numfilters


    conv1_shape = (conv1filter, conv1filter, num_channels, conv1numfilters)
    conv2_shape = (conv2filter, conv2filter, conv1numfilters, conv2numfilters)

    with tf.variable_scope(scope):
        normal_init = tf.truncated_normal_initializer(mean=0.0, stddev=FLAGS.initstd)
        return [tf.get_variable(name='conv1w', shape=conv1_shape, initializer=normal_init),
             tf.get_variable(name='conv1b', shape=[conv1numfilters], initializer=tf.constant_initializer(0.0)),
             tf.get_variable(name='conv2w', shape=conv2_shape, initializer=normal_init),
             tf.get_variable(name='conv2b', shape=[conv2numfilters], initializer=tf.constant_initializer(0.0)),
             tf.get_variable(name='1w', shape=[flat_dim, l1], initializer=normal_init),
             tf.get_variable(name='1b', shape=[l1], initializer=tf.constant_initializer(0.0)),
             tf.get_variable(name='2w', shape=[l1, dimOut], initializer=normal_init),
             tf.get_variable(name='2b', shape=[dimOut], initializer=tf.constant_initializer(0.0))]


def build_convnet_layers(x, theta):
    # h1 = tf.matmul(x, theta[0]) + theta[1]
    # h1 = tf.nn.relu(h1)
    # h2 = tf.matmul(h1, theta[2]) + theta[3]
    # h2 = tf.nn.relu(h2)
    # h3 = tf.matmul(h2, theta[4]) + theta[5]
    # return h3
    h0 = tf.identity(x, name='h0-obs')
    h0r = tf.reshape(h0, [-1, height, width, num_channels])
    h1 = tf.nn.relu(tf.nn.conv2d(h0r, theta[0], strides=strides1, padding=padding) + theta[1], name='conv1')
    # if FLAGS.batchnorm == True:
    #     x = batch_norm(h1, scope='b1')
    # else:
    #     x = h1
    h2 = tf.nn.relu(tf.nn.conv2d(h1, theta[2], strides=strides2, padding=padding) + theta[3], name='conv2')
    h2_flat = tf.reshape(h2, [-1, flat_dim])
    h3 = tf.nn.relu(tf.matmul(h2_flat, theta[4]) + theta[5], name='h1')
    h4 = tf.matmul(h3, theta[6]) + theta[7]
    return h4

def lfunction(obs, theta, scope="lfunction"):
    with tf.variable_scope(scope):
        l = build_convnet_layers(obs, theta)
        return l


def vec2trimat(vec, dim):
    L = tf.reshape(vec, [-1, dim, dim])
    L = tf.batch_matrix_band_part(L, -1, 0) - tf.batch_matrix_diag(tf.batch_matrix_diag_part(L)) + \
        tf.batch_matrix_diag(tf.exp(tf.batch_matrix_diag_part(L)))
    return L


def ufunction(obs, theta, scope="ufunction"):
    with tf.variable_scope(scope):
        act = build_convnet_layers(obs, theta)
        act = tf.tanh(act)
        return act


def afunction(action, lvalue, uvalue, dimA, scope="afunction"):
    with tf.variable_scope(scope):
        delta = action - uvalue
        L = vec2trimat(lvalue, dimA)

        h1 = tf.reshape(delta, [-1, 1, dimA])
        h1 = tf.batch_matmul(h1, L)  # batch:1:dimA
        h1 = tf.squeeze(h1, [1])  # batch:dimA
        h2 = -tf.constant(0.5) * tf.reduce_sum(h1 * h1, 1)  # batch

        return h2


def qfunction(obs, avalue, theta, scope="qfunction"):
    with tf.variable_scope(scope):
        q = build_convnet_layers(obs, theta)
        q = tf.squeeze(q, [1]) + avalue
        return q
