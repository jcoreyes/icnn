# Code from Repo SimonRamstedt/ddpg
# Heavily modified

import os

import numpy as np
import tensorflow as tf

import ddpg_nets_dm
import ddpg_convnets_dm
from replay_memory import ReplayMemory

flags = tf.app.flags
FLAGS = flags.FLAGS


# DDPG Agent
class Actor(object):
    def __init__(self, use_conv, nets, dimO, dimA, obs, obs2, is_training, sess, scope='actor'):
        self.use_conv = use_conv
        self.nets = nets
        self.dimO = dimO
        self.dimA = dimA
        self.scope = scope
        self.summary_list = []

        if use_conv:
            self.theta_p = nets.theta_p(dimO, dimA, FLAGS.conv1filter, FLAGS.conv1numfilters,
                                        FLAGS.conv2filter, FLAGS.conv2numfilters, FLAGS.l1size,
                                        FLAGS.l2size)

        else:
            self.theta_p = nets.theta_p(dimO, dimA, FLAGS.l1size, FLAGS.l2size)
        self.theta_pt, self.update_pt = exponential_moving_averages(self.theta_p, FLAGS.tau)
       
        self._create_policy_ops(obs, obs2, is_training)
        self._create_policy_funs(obs, sess, is_training)

    def _create_policy_ops(self, obs, obs2, is_training):
        with tf.variable_scope(self.scope):
            # Will be used by critic to compute policy gradient
            self.train_policy = self.nets.policy(obs, self.theta_p, is_training, reuse=None)
            self.test_policy = self.nets.policy(obs, self.theta_p, is_training, reuse=True)
            self.explore_policy = self.test_policy + self._compute_noise()

            self.act2 = self.nets.policy(obs2, self.theta_pt, is_training, reuse=True)

    def _create_policy_funs(self, obs, sess, is_training):
        # Create functions for execution
        with sess.as_default():
            self._act_test = Fun([obs, is_training], self.test_policy)
            self._act_expl = Fun([obs, is_training], self.explore_policy)
            self._reset = Fun([], self.ou_reset)


    def _compute_noise(self):
        noise_init = tf.zeros([1] + self.dimA)
        noise_var = tf.Variable(noise_init)
        self.ou_reset = noise_var.assign(noise_init)
        noise = noise_var.assign_sub((FLAGS.outheta) * noise_var
                                     - tf.random_normal(self.dimA, stddev=FLAGS.ousigma))
        return noise 
 
    def compute_loss(self, critic, obs, obs2, is_training):
        # Used for computing policy gradient. Q(obs, train_policy)
        with tf.variable_scope(critic.scope):
            # Compute Q Target
            self.q2 = self.nets.qfunction(obs2, self.act2, critic.theta_qt, is_training, reuse=True)
            self.q_train_policy = self.nets.qfunction(obs, self.train_policy, critic.theta_q,
                                                 is_training, reuse=True)

        # Policy loss
        meanq = tf.reduce_mean(self.q_train_policy, 0)
        wd_p = tf.add_n([FLAGS.pl2norm * tf.nn.l2_loss(var) for var in self.theta_p])  # weight decay
        loss_p = -meanq + wd_p

        # Policy optimization
        optim_p = tf.train.AdamOptimizer(learning_rate=FLAGS.prate)
        grads_and_vars_p = optim_p.compute_gradients(loss_p, var_list=self.theta_p)
        optimize_p = optim_p.apply_gradients(grads_and_vars_p)
        with tf.control_dependencies([optimize_p]):
            self.train_p = tf.group(self.update_pt)

    def act(self, obs, test):
        if test:
            return self._act_test(obs, False)
        else:
            return self._act_expl(obs, True)

    def reset(self):
        self._reset()

    def get_summary(self):
        return self.summary_list

    def get_train_outputs(self):
        return [self.train_p]

class Critic(object):
    def __init__(self, use_conv, nets, dimO, dimA, obs, is_training, act_train, scope='critic'):
        self.use_conv = use_conv
        self.nets = nets
        self.dimO = dimO
        self.dimA = dimA
        self.scope = scope
        self.summary_list = []

        if use_conv:
            self.theta_q = nets.theta_q(dimO, dimA, FLAGS.conv1filter, FLAGS.conv1numfilters,
                                        FLAGS.conv2filter, FLAGS.conv2numfilters, FLAGS.l1size,
                                        FLAGS.l2size)
        else:
            self.theta_q = nets.theta_q(dimO, dimA, FLAGS.l1size, FLAGS.l2size)
        self.theta_qt, self.update_qt = exponential_moving_averages(self.theta_q, FLAGS.tau)

        with tf.variable_scope(self.scope):
            self.q_train = nets.qfunction(obs, act_train, self.theta_q, is_training, reuse=None)

    def compute_loss(self, rew, term2, q2):

            # Used for computing TD error
            self.q_target = tf.stop_gradient(tf.select(term2, rew, rew + FLAGS.discount * q2))

            # q loss
            td_error = self.q_train - self.q_target
            ms_td_error = tf.reduce_mean(tf.square(td_error), 0)
            wd_q = tf.add_n([FLAGS.l2norm * tf.nn.l2_loss(var) for var in self.theta_q])  # weight decay
            self.loss_q = ms_td_error + wd_q

            # q optimization
            optim_q = tf.train.AdamOptimizer(learning_rate=FLAGS.rate)
            grads_and_vars_q = optim_q.compute_gradients(self.loss_q, var_list=self.theta_q)
            optimize_q = optim_q.apply_gradients(grads_and_vars_q)
            with tf.control_dependencies([optimize_q]):
                self.train_q = tf.group(self.update_qt)

    def get_summary(self):
        return self.summary_list

    def get_train_outputs(self):
        return [self.train_q, self.loss_q]

class Agent(object):

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


        # Placeholders
        input_obs_dim = [None] + dimO

        obs = tf.placeholder(tf.float32, input_obs_dim, "obs")
        is_training = tf.placeholder(tf.bool, [], name='is_training')
        act_train = tf.placeholder(tf.float32, [FLAGS.bsize] + dimA, "act_train")
        rew = tf.placeholder(tf.float32, [FLAGS.bsize], "rew")
        obs2 = tf.placeholder(tf.float32, [FLAGS.bsize] + dimO, "obs2")
        term2 = tf.placeholder(tf.bool, [FLAGS.bsize], "term2")

        summary_writer = tf.train.SummaryWriter(os.path.join(FLAGS.outdir, 'board'), self.sess.graph)
        summary_list = []
        summary_list.append(tf.scalar_summary('reward', tf.reduce_mean(rew)))

        self.setup_actor_critic(nets, dimO, dimA, obs, obs2, is_training, rew, term2, act_train)
        summary_list.extend(self.actor.get_summary())
        summary_list.extend(self.critic.get_summary())

        # summary_list.append(tf.scalar_summary('Qvalue', tf.reduce_mean(q_train)))
        # summary_list.append(tf.scalar_summary('loss', ms_td_error))

        # tf functions
        with self.sess.as_default():
            train_inputs = [obs, act_train, rew, obs2, term2, is_training]
            train_outputs = self.actor.get_train_outputs() + self.critic.get_train_outputs()
            self._train = Fun(train_inputs,
                              train_outputs,
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

    def setup_actor_critic(self, nets, dimO, dimA, obs, obs2, is_training, rew, term2, act_train):
        self.actor = Actor(self.use_conv, nets, dimO, dimA, obs, obs2, is_training,
                           self.sess, scope='actor')
        self.critic = Critic(self.use_conv, nets, dimO, dimA, obs, is_training, act_train,
                             scope='critic')

        self.actor.compute_loss(self.critic, obs, obs2, is_training)
        self.critic.compute_loss(rew, term2, self.actor.q2)


    def reset(self, obs):
        self.actor.reset()
        self.observation = obs  # initial observation

    def act(self, test=False):
        obs = np.expand_dims(self.observation, axis=0)
        action = self.actor.act(obs, test)
        action = np.clip(action, -1, 1)
        self.action = np.atleast_1d(np.squeeze(action, axis=0))  # TODO: remove this hack
        return self.action

    def observe(self, rew, term, obs2, test=False):

        obs1 = self.observation
        self.observation = obs2

        # train
        if not test:
            self.t = self.t + 1
            self.rm.enqueue(obs1, term, self.action, rew)

            if self.t > FLAGS.warmup:
                for i in xrange(FLAGS.iter):
                    loss = self.train()

    def train(self):
        obs, act, rew, ob2, term2, info = self.rm.minibatch(size=FLAGS.bsize)
        _, _, loss = self._train(obs, act, rew, ob2, term2, True, log=FLAGS.summary, global_step=self.t)
        return loss

    def __del__(self):
        self.sess.close()



# Tensorflow utils
#
class Fun:
    """ Creates a python function that maps between inputs and outputs in the computational graph. """

    def __init__(self, inputs, outputs, summary_ops=None, summary_writer=None, session=None):
        self._inputs = inputs if type(inputs) == list else [inputs]
        self._outputs = outputs
        self._summary_op = tf.merge_summary(summary_ops) if type(summary_ops) == list else summary_ops
        self._session = session or tf.get_default_session()
        self._writer = summary_writer

    def __call__(self, *args, **kwargs):
        """
        Arguments:
          **kwargs: input values
          log: if True write summary_ops to summary_writer
          global_step: global_step for summary_writer
        """
        log = kwargs.get('log', False)

        feeds = {}
        for (argpos, arg) in enumerate(args):
            feeds[self._inputs[argpos]] = arg

        out = self._outputs + [self._summary_op] if log else self._outputs
        res = self._session.run(out, feeds)

        if log:
            i = kwargs['global_step']
            self._writer.add_summary(res[-1], global_step=i)
            res = res[:-1]

        return res

def exponential_moving_averages(theta, tau=0.001):
    ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
    update = ema.apply(theta)  # also creates shadow vars
    averages = [ema.average(x) for x in theta]
    return averages, update
