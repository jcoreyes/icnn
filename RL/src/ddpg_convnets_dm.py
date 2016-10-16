# Code from Repo SimonRamstedt/ddpg
# Heavily modified

import numpy as np
import tensorflow as tf


def fanin_init(shape, fanin=None):
    fanin = fanin or shape[0]
    v = 1 / np.sqrt(fanin)
    return tf.random_uniform(shape, minval=-v, maxval=v)

flat_dim = 8*8*32
padding = 'SAME'
num_channels = 3
width, height = 32, 32
strides = [1, 2, 2, 1]
def theta_p(dimO, dimA, conv1filter, conv1numfilters, conv2filter, conv2numfilters, l1, l2):
    dimO = dimO[0]
    dimA = dimA[0]
    conv1_shape = (conv1filter, conv1filter, num_channels, conv1numfilters)
    conv2_shape = (conv2filter, conv2filter, conv1numfilters, conv2numfilters)
    with tf.variable_scope("theta_p"):
        return [tf.Variable(tf.random_uniform(conv1_shape, -3e-3, 3e-3), name='conv1'),
                tf.Variable(tf.random_uniform(conv2_shape, -3e-3, 3e-3), name='conv2'),
                tf.Variable(fanin_init([flat_dim, l1]), name='1w'),
                tf.Variable(fanin_init([l1], flat_dim), name='1b'),
                tf.Variable(fanin_init([l1, l2]), name='2w'),
                tf.Variable(fanin_init([l2], l1), name='2b'),
                tf.Variable(tf.random_uniform([l2, dimA], -3e-3, 3e-3), name='3w'),
                tf.Variable(tf.random_uniform([dimA], -3e-3, 3e-3), name='3b')]


def policy(obs, theta, name='policy'):
    with tf.variable_op_scope([obs], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h0r = tf.reshape(h0, [-1, height, width, num_channels])
        h1 = tf.nn.relu(tf.nn.conv2d(h0r, theta[0], strides=strides, padding=padding), name='conv1')
        h2 = tf.nn.relu(tf.nn.conv2d(h1, theta[1], strides=strides, padding=padding), name='conv2')
        h2_flat = tf.reshape(h2, [-1, flat_dim])
        h3 = tf.nn.relu(tf.matmul(h2_flat, theta[2]) + theta[3], name='h1')
        h4 = tf.nn.relu(tf.matmul(h3, theta[4]) + theta[5], name='h2')
        h5 = tf.identity(tf.matmul(h4, theta[6]) + theta[7], name='h3')
        action = tf.nn.tanh(h5, name='h4-action')
        return action


def theta_q(dimO, dimA, conv1filter, conv1numfilters, conv2filter, conv2numfilters, l1, l2):
    dimO = dimO[0]
    dimA = dimA[0]
    conv1_shape = (conv1filter, conv1filter, num_channels, conv1numfilters)
    conv2_shape = (conv2filter, conv2filter, conv1numfilters, conv2numfilters)
    with tf.variable_scope("theta_q"):
        return [tf.Variable(tf.random_uniform(conv1_shape, -3e-3, 3e-3), name='conv1'),
                tf.Variable(tf.random_uniform(conv2_shape, -3e-3, 3e-3), name='conv2'),
                tf.Variable(fanin_init([flat_dim, l1]), name='1w'),
                tf.Variable(fanin_init([l1], flat_dim), name='1b'),
                tf.Variable(fanin_init([l1 + dimA, l2]), name='2w'),
                tf.Variable(fanin_init([l2], l1 + dimA), name='2b'),
                tf.Variable(tf.random_uniform([l2, 1], -3e-3, 3e-3), name='3w'),
                tf.Variable(tf.random_uniform([1], -3e-4, 3e-4), name='3b')]


def qfunction(obs, act, theta, name="qfunction"):
    with tf.variable_op_scope([obs, act], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        # h0a = tf.identity(act, name='h0-act')
        h0r = tf.reshape(h0, [-1, height, width, num_channels])
        h1 = tf.nn.relu(tf.nn.conv2d(h0r, theta[0], strides=strides, padding=padding), name='conv1')
        h2 = tf.nn.relu(tf.nn.conv2d(h1, theta[1], strides=strides, padding=padding), name='conv2')
        h2_flat = tf.reshape(h2, [-1, flat_dim])
        h3 = tf.nn.relu(tf.matmul(h2_flat, theta[2]) + theta[3], name='h1')

        h1a = tf.concat(1, [h3, act])
        h2 = tf.nn.relu(tf.matmul(h1a, theta[4]) + theta[5], name='h2')
        qs = tf.matmul(h2, theta[6]) + theta[7]
        q = tf.squeeze(qs, [1], name='h3-q')
        return q
