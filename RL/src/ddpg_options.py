# Code from Repo SimonRamstedt/ddpg
# Heavily modified

import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.bayesflow as bf
import ddpg_nets_dm
import ddpg_convnets_dm
from replay_memory import ReplayMemory
from ddpg import Actor, Critic, Agent, Fun, exponential_moving_averages
flags = tf.app.flags
FLAGS = flags.FLAGS


# DDPG Agent

class HyperOptionsActor(Actor):
    def __init__(self, use_conv, nets, dimO, dimA, obs, obs2, is_training,
                 sess, scope='hyperactor'):
        self.actors = []
        with tf.variable_scope(scope):
            for i in range(FLAGS.num_options):
                actor = Actor(use_conv, nets, dimO, dimA, obs, obs2, is_training,
                              sess, scope='actor%d' % i)
                self.actors.append(actor)

        super(HyperOptionsActor, self).__init__(use_conv, nets, dimO, [FLAGS.num_options], obs, obs2,
                                                is_training, sess, scope)

    def _create_policy_ops(self, obs, obs2, is_training):
        with tf.variable_scope(self.scope):
            # Will be used by critic to compute policy gradient
            self.train_policy = self.nets.policy(obs, self.theta_p, is_training, reuse=None, l1_act=tf.nn.softmax)
            self.test_policy = self.nets.policy(obs, self.theta_p, is_training, reuse=True, l1_act=tf.nn.softmax)
            self.explore_policy = self.test_policy #+ self._compute_noise()

            self.options2 = self.nets.policy(obs2, self.theta_pt, is_training, reuse=True, l1_act=tf.nn.softmax)
            self.act2 = gather_cols(tf.concat(1, [actor.act2 for actor in self.actors]),
                                    tf.to_int32(tf.argmax(self.options2, 0)))
            # Action explore
            options_explore = tf.concat(1, [actor.explore_policy for actor in self.actors])
            # options, options_prob = self.compute_options(obs, nets, is_training, reuse=True)
            self.act_expl = gather_cols(options_explore, tf.to_int32(tf.argmax(self.test_policy, 0)))
            self.act_test = self.act_expl

    def _create_policy_funs(self, obs, sess, is_training):
        # Create functions for execution
        with sess.as_default():
            self._act_test = Fun([obs, is_training], self.act_test)
            self._act_expl = Fun([obs, is_training], self.act_expl)

    def compute_loss(self, critic, obs, is_training):
        for i, actor in enumerate(self.actors):
            actor.compute_loss(critic, obs, is_training)

        # Q value using each option
        self.q_values = tf.pack([actor.q_train_policy for actor in self.actors], 1)

        weighted_q_values = tf.reduce_mean(tf.mul(self.train_policy, self.q_values), 1)
        meanq = tf.reduce_mean(weighted_q_values, 0) #[tf.reduce_mean(q_value, 0) for q_value in q_values]
        entropy_reg = FLAGS.entropyreg * tf.reduce_mean(entropy(self.train_policy), 0)

        wd_p = tf.add_n([FLAGS.pl2norm * tf.nn.l2_loss(var) for var in self.theta_p])  # weight decay
        loss_p = -meanq + wd_p + -entropy_reg

        # Policy optimization
        optim_p = tf.train.AdamOptimizer(learning_rate=FLAGS.prate)
        grads_and_vars_p = optim_p.compute_gradients(loss_p, var_list=self.theta_p)
        optimize_p = optim_p.apply_gradients(grads_and_vars_p)
        with tf.control_dependencies([optimize_p]):
            self.train_hyper_p = tf.group(self.update_pt)

        self.train_p = [self.train_hyper_p] + [actor.train_p for actor in self.actors]

    def reset(self):
        for actor in self.actors:
            actor._reset()


    def get_summary(self):
        self.summary_list.append(tf.histogram_summary('options_prob', self.explore_policy))
        for i, actor in enumerate(self.actors):
            self.summary_list.append(tf.histogram_summary('actor_%d_action' %i, actor.explore_policy))

        return self.summary_list

    def get_train_outputs(self):
        return self.train_p

class OptionsAgent(Agent):
    def __init__(self, dimO, dimA):
        super(OptionsAgent, self).__init__(dimO, dimA)

    def setup_actor_critic(self, nets, dimO, dimA, obs, obs2, is_training, rew, term2, act_train):
        self.actor = HyperOptionsActor(self.use_conv, nets, dimO, dimA, obs, obs2, is_training,
                           self.sess, scope='actor')
        self.critic = Critic(self.use_conv, nets, dimO, dimA, obs, obs2, rew, term2, is_training,
                             act_train, self.actor, scope='critic')

        self.actor.compute_loss(self.critic, obs, is_training)

    def train(self):
        obs, act, rew, ob2, term2, info = self.rm.minibatch(size=FLAGS.bsize)
        output = self._train(obs, act, rew, ob2, term2, True, log=FLAGS.summary, global_step=self.t)
        loss = output[-1]
        return loss

def entropy(p):
    return - tf.reduce_sum(p * tf.log(p), 1)

def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.

    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], -1])
