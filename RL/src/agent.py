import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('warmup', 1000, 'time without training but only filling the replay memory')
flags.DEFINE_integer('bsize', 256, 'minibatch size')
flags.DEFINE_integer('iter', 1, 'train iters each timestep')
flags.DEFINE_integer('l1size', 256, '1st layer size')
flags.DEFINE_integer('l2size', 200, '2nd layer size')
flags.DEFINE_integer('conv1filter', 4, 'Conv1 filter size')
flags.DEFINE_integer('conv1numfilters', 16, "Number of conv1 filters")
flags.DEFINE_integer('conv2filter', 4, 'Conv2 filter size')
flags.DEFINE_integer('conv2numfilters', 32, "Number of conv2 filters")
flags.DEFINE_boolean('batchnorm', False, 'Use batch norm after conv layers')

flags.DEFINE_integer('rmsize', 500000, 'memory size')

flags.DEFINE_float('tau', 0.01, 'moving average for target network')
flags.DEFINE_float('discount', 0.99, '')
flags.DEFINE_float('l2norm', 0.0001, 'l2 weight decay')
flags.DEFINE_float('pl2norm', 0., 'policy network l2 weight decay (only for DDPG)')
flags.DEFINE_float('rate', 0.001, 'learning rate')
flags.DEFINE_float('prate', 0.0001, 'policy net learning rate (only for DDPG)')
flags.DEFINE_float('outheta', 0.15, 'noise theta')
flags.DEFINE_float('ousigma', 0.1, 'noise sigma')
flags.DEFINE_float('lrelu', 0.01, 'leak relu rate')
flags.DEFINE_float('entropyreg', 0.0001, 'entropy regularization on options (only for DDPGOptions')

flags.DEFINE_integer('thread', 1, 'tensorflow threads')



flags.DEFINE_float('initstd', 0.01, 'weight init std (DDPG uses its own initialization)')
