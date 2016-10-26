import tensorflow as tf
import sys
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('runs', 10, 'total runs')
flags.DEFINE_integer('total', 100000, 'total timesteps')
flags.DEFINE_integer('train', 1000, 'training timesteps between testing')
flags.DEFINE_string('data', '.', 'dir contains outputs of DDPG, NAF and ICNN')


import matplotlib.pyplot as plt
import numpy as np

plt.style.use('bmh')
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
plt.xlabel('Timestep')
plt.ylabel('Reward')



def get_data(log_file):
    with open(log_file) as f:
        lines = [line.split() for line in f.readlines()[1:]]
        data = np.asfarray(lines)
        steps, avg_r, std_r, min_r, max_r = [data[:, i] for i in range(5)]
    return steps, avg_r, std_r, min_r, max_r




if __name__ == '__main__':
    steps, avg_r, std_r, min_r, max_r = get_data(FLAGS.data)
    plt.plot(steps, avg_r)
    plt.fill_between(steps, min_r, max_r, alpha=0.1)
    plt.title("Avg reward")
    plt.savefig(os.path.join(os.path.dirname(FLAGS.data), 'reward_plot.png'))
    plt.show()

    # lines = []
    # for name, item, color in zip(names, folders, colors):
    #     X, Ymin, Ymax, Ymean = get_data(item, FLAGS.total / FLAGS.train, FLAGS.train)
    #     line, = plt.plot(X, Ymean, label=name, color=color)
    #     lines.append(line)
    #     plt.fill_between(X, Ymin, Ymax, alpha=0.1, color=color)
    # plt.legend(handles=lines, loc=2)
    # plt.savefig(FLAGS.data + '/result.pdf')
