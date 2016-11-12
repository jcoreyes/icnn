# Code from Repo SimonRamstedt/ddpg
# Heavily modified

import numpy as np
import tensorflow as tf
#from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.slim import batch_norm
from ddpg_convnets_dm import fanin_init, conv, mlp
flags = tf.app.flags
FLAGS = flags.FLAGS
padding = 'SAME'
num_channels = 3 * FLAGS.num_frames
width, height = FLAGS.width, FLAGS.height
flat_dim = 8*8*32
strides1 = [1, 2, 2, 1]
strides2 = [1, 2, 2, 1]

def theta_p(dimO, dimA, conv1filter, conv1numfilters, conv2filter, conv2numfilters, l1, l2):
    dimO = dimO[0]
    dimA = (dimA[0] - FLAGS.num_options) / FLAGS.num_options 
    conv1_shape = (conv1filter, conv1filter, num_channels, conv1numfilters)
    conv2_shape = (conv2filter, conv2filter, conv1numfilters, conv2numfilters)
    with tf.variable_scope("theta_p"):
        vars = [tf.Variable(tf.random_uniform(conv1_shape, -3e-3, 3e-3), name='conv1w'),
                tf.Variable(tf.constant(0.1, shape=[conv1numfilters]), name='conv1b'),
                tf.Variable(tf.random_uniform(conv2_shape, -3e-3, 3e-3), name='conv2w'),
                tf.Variable(tf.constant(0.1, shape=[conv2numfilters]), name='conv2b'),
                tf.Variable(fanin_init([flat_dim, l1]), name='1w'),
                tf.Variable(fanin_init([l1], flat_dim), name='1b')]

        for i in range(FLAGS.num_options):
            vars.extend([tf.Variable(tf.random_uniform([l1, dimA], -3e-3, 3e-3), name='action%d_2w' %i),
                         tf.Variable(tf.random_uniform([dimA], -3e-3, 3e-3), name='action%d_2b' %i)])
        vars.extend([tf.Variable(tf.random_uniform([l1, FLAGS.num_options], -3e-3, 3e-3), name='opt%d_2w' %i),
                     tf.Variable(tf.random_uniform([FLAGS.num_options], -3e-3, 3e-3), name='opt%d_2b' %i)])

        return vars


def policy(obs, theta, is_training, reuse=False, name='policy', l1_act=tf.nn.tanh):
    with tf.variable_op_scope([obs], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        # tensorflow expects (batch_size, height, width, num_channels)
        #h0r = tf.reshape(h0, [-1, height, width, num_channels])
        h1 = conv(h0, theta[0], theta[1], strides1, padding, is_training, reuse, name +'bn1', 'conv1')
        h2 = conv(h1, theta[2], theta[3], strides2, padding, is_training, reuse, name + 'bn2', 'conv2')
        h2_flat = tf.reshape(h2, [-1, flat_dim])
        h3 = mlp(h2_flat, theta[4], theta[5], tf.nn.relu, 'h1')

        s = 6
        actions = [mlp(h3, theta[s+i], theta[s+i+1], tf.nn.tanh, 'h2%d' %(s+i)) for i in range(0, 2*FLAGS.num_options, 2)]
        options = mlp(h3, theta[-2], theta[-1], tf.nn.softmax, 'options')
        output = actions + [options]#tf.concat(1, actions + [options])
        return (actions, options)


def theta_q(dimO, dimA, conv1filter, conv1numfilters, conv2filter, conv2numfilters, l1, l2):
    dimO = dimO[0]
    dimA = dimA[0]
    conv1_shape = (conv1filter, conv1filter, num_channels, conv1numfilters)
    conv2_shape = (conv2filter, conv2filter, conv1numfilters, conv2numfilters)
    with tf.variable_scope("theta_q"):
        return [tf.Variable(tf.random_uniform(conv1_shape, -3e-3, 3e-3), name='conv1w'),
                tf.Variable(tf.constant(0.1, shape=[conv1numfilters]), name='conv1b'),
                tf.Variable(tf.random_uniform(conv2_shape, -3e-3, 3e-3), name='conv2w'),
                tf.Variable(tf.constant(0.1, shape=[conv2numfilters]), name='conv2b'),
                tf.Variable(fanin_init([flat_dim, l1]), name='1w'),
                tf.Variable(fanin_init([l1], flat_dim), name='1b'),
                tf.Variable(fanin_init([l1 + dimA, l2]), name='2w'),
                tf.Variable(fanin_init([l2], l1 + dimA), name='2b'),
                tf.Variable(tf.random_uniform([l2, 1], -3e-3, 3e-3), name='3w'),
                tf.Variable(tf.random_uniform([1], -3e-3, 3e-3), name='3b')]

def qfunction(obs, act, theta, is_training, reuse=False, name="qfunction"):
    with tf.variable_op_scope([obs, act], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        # h0a = tf.identity(act, name='h0-act')
       # h0r = tf.reshape(h0, [-1, height, width, num_channels])
        h1 = conv(h0, theta[0], theta[1], strides1, padding, is_training, reuse, name + 'bn1', 'conv1')
        h2 = conv(h1, theta[2], theta[3], strides2, padding, is_training, reuse, name + 'bn2', 'conv2')
        h2_flat = tf.reshape(h2, [-1, flat_dim])
        h3 = mlp(h2_flat, theta[4], theta[5], tf.nn.relu, 'h1')
        h1a = tf.concat(1, [h3, act])
        h2a = mlp(h1a, theta[6], theta[7], tf.nn.relu, 'h2')
        qs = tf.matmul(h2a, theta[8]) + theta[9]
        q = tf.squeeze(qs, [1], name='h3-q')
        return q
