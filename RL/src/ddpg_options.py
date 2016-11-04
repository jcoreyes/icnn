# Code from Repo SimonRamstedt/ddpg
# Heavily modified

import os

import numpy as np
import tensorflow as tf

import ddpg_nets_dm
import ddpg_convnets_dm
from replay_memory import ReplayMemory
from ddpg import Actor, Critic, Agent, Fun
flags = tf.app.flags
FLAGS = flags.FLAGS


# DDPG Agent

class HyperOptionsActor(Actor):
    def __init__(self, use_conv, nets, dimO, dimA, scope='hyperactor'):
        super(HyperOptionsActor, self).__init__(use_conv, nets, dimO, dimA, scope)
        self.actors = [Actor(use_conv, nets, dimO, dimA) for _ in range(FLAGS.num_options)]

    def compute_options(self, obs, nets, is_training, reuse):
        with tf.variable_scope(self.scope):
            options_prob = nets.policy(obs, self.theta_p, is_training, reuse=reuse)
        options = [actor.act_train_policy for actor in self.actors]
        return options, options_prob

    def compute_loss(self, nets, obs, dimA, is_training, critic, sess):
        for i, actor in enumerate(self.actors):
            with tf.variable_scope('actor%d/' %i):
                actor.compute_loss(nets, obs, dimA, is_training, critic, sess)

        # Policy loss
        options, options_prob = self.compute_options(obs, nets, is_training, reuse=None)
        q_values = tf.pack([actor.q_train_policy for actor in self.actors], 1)

        weighted_q_values = tf.reduce_mean(tf.mul(options_prob, q_values), 1)
        meanq = tf.reduce_mean(weighted_q_values, 0) #[tf.reduce_mean(q_value, 0) for q_value in q_values]

        wd_p = tf.add_n([FLAGS.pl2norm * tf.nn.l2_loss(var) for var in self.theta_p])  # weight decay
        loss_p = -meanq + wd_p

        # Policy optimization
        optim_p = tf.train.AdamOptimizer(learning_rate=FLAGS.prate)
        grads_and_vars_p = optim_p.compute_gradients(loss_p, var_list=self.theta_p)
        optimize_p = optim_p.apply_gradients(grads_and_vars_p)
        with tf.control_dependencies([optimize_p]):
            self.train_hyper_p = tf.group(self.update_pt)

        self.train_p = [self.train_hyper_p] + [actor.train_p for actor in self.actors]


        # Action explore
        with tf.variable_scope(self.scope):
            options_prob = nets.policy(obs, self.theta_p, is_training, reuse=True)
        options = tf.concat(1, [actor.act_expl for actor in self.actors])
        #options, options_prob = self.compute_options(obs, nets, is_training, reuse=True)
        self.act_expl = gather_cols(options, tf.to_int32(tf.argmax(options_prob, 0)))
        self.act_test = self.act_expl

        # Create functions for execution
        with sess.as_default():
            self._act_test = Fun([obs, is_training], self.act_test)
            self._act_expl = Fun([obs, is_training], self.act_expl)

    def reset(self):
        for actor in self.actors:
            actor._reset()

class OptionsAgent(Agent):
    def __init__(self, dimO, dimA):
        dimA = list(dimA)
        dimO = list(dimO)
        if len(dimO) > 1:
            assert len(dimO) == 3
            self.use_conv = True
            nets = ddpg_convnets_dm
        else:
            self.use_conv = False
            nets = ddpg_nets_dm

        # init replay memory
        self.rm = ReplayMemory(FLAGS.rmsize, dimO, dimA)
        # start tf session
        self.sess = tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=FLAGS.thread,
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)))


        input_obs_dim = [None] + dimO

        obs = tf.placeholder(tf.float32, input_obs_dim, "obs")
        is_training = tf.placeholder(tf.bool, [], name='is_training')
        act_train = tf.placeholder(tf.float32, [FLAGS.bsize] + dimA, "act_train")
        rew = tf.placeholder(tf.float32, [FLAGS.bsize], "rew")
        obs2 = tf.placeholder(tf.float32, [FLAGS.bsize] + dimO, "obs2")
        term2 = tf.placeholder(tf.bool, [FLAGS.bsize], "term2")

        self.actor = HyperOptionsActor(self.use_conv, nets, dimO, dimA)
        self.critic = Critic(self.use_conv, nets, dimO, dimA)

        self.actor.compute_loss(nets, obs, dimA, is_training, self.critic, self.sess)
        self.critic.compute_loss(nets, obs, is_training, act_train, rew, obs2, term2, self.actor)

        summary_writer = tf.train.SummaryWriter(os.path.join(FLAGS.outdir, 'board'), self.sess.graph)
        summary_list = []
        # summary_list.append(tf.scalar_summary('Qvalue', tf.reduce_mean(q_train)))
        # summary_list.append(tf.scalar_summary('loss', ms_td_error))
        summary_list.append(tf.scalar_summary('reward', tf.reduce_mean(rew)))

        # tf functions
        with self.sess.as_default():
            self.outputs = self.actor.train_p + [self.critic.train_q, self.critic.loss_q]
            self._train = Fun([obs, act_train, rew, obs2, term2, is_training],
                              self.outputs,
                              summary_list, summary_writer)

        # initialize tf variables
        self.saver = tf.train.Saver(max_to_keep=1)
        ckpt = tf.train.latest_checkpoint(FLAGS.outdir + "/tf")
        if ckpt:
            self.saver.restore(self.sess, ckpt)
        else:
            self.sess.run(tf.initialize_all_variables())

        self.sess.graph.finalize()

        self.t = 0  # global training time (number of observations)


    def train(self):
        obs, act, rew, ob2, term2, info = self.rm.minibatch(size=FLAGS.bsize)
        output = self._train(obs, act, rew, ob2, term2, True, log=FLAGS.summary, global_step=self.t)
        loss = output[-1]
        return loss

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
