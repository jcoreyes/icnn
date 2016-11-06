# Code from Repo SimonRamstedt/ddpg
# Heavily modified

import os
import pprint

import gym
import numpy as np
import tensorflow as tf

import agent
import sys
import subprocess
import normalized_env
import runtime_env
from malmo import Minecraft


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('env', '', 'gym environment')
flags.DEFINE_string('outdir', 'output', 'output directory')
flags.DEFINE_boolean('force', False, 'overwrite existing results')
flags.DEFINE_integer('train', 1000, 'training timesteps between testing episodes')
flags.DEFINE_integer('test', 10, 'testing episodes between training timesteps')
flags.DEFINE_integer('tmax', 1000, 'maxium timesteps each episode')
flags.DEFINE_integer('total', 2e4, 'total training timesteps')
flags.DEFINE_float('monitor', 0.01, 'probability of monitoring a test episode')
flags.DEFINE_string('model', 'DDPG', 'reinforcement learning model[DDPG, NAF, ICNN]')
flags.DEFINE_integer('tfseed', 0, 'random seed for tensorflow')
flags.DEFINE_integer('gymseed', 0, 'random seed for openai gym')
flags.DEFINE_integer('npseed', 0, 'random seed for numpy')
flags.DEFINE_boolean('summary', True, 'where to do tensorboard summmary')

# Option specific
flags.DEFINE_integer('num_options', 2, 'Only applies to DDPGOptions.')
flags.DEFINE_boolean('stochastic_options', True, 'Choose options stochastically')

# Env specific
flags.DEFINE_boolean('vision', True, 'whether to use vision observations')
flags.DEFINE_integer('width', 32, 'width of video obs')
flags.DEFINE_integer('height', 32, 'height of video obs')
flags.DEFINE_string('maze', 'TMaze', 'type of maze')
flags.DEFINE_boolean('reset', False, 'whether to recreate minecraft env each time')
flags.DEFINE_integer('num_parallel', 1, 'how many servers to use at same time.')


if FLAGS.model == 'DDPG':
    import ddpg
    Agent = ddpg.Agent
if FLAGS.model == 'DDPGOptions':
    import ddpg_options
    Agent = ddpg_options.OptionsAgent
elif FLAGS.model == 'NAF':
    import naf
    Agent = naf.Agent
elif FLAGS.model == 'ICNN':
    import icnn
    Agent = icnn.Agent


class Experiment(object):

    def run(self):
        self.train_timestep = 0
        self.test_timestep = 0

        # create normal
        maze_def = {'type': FLAGS.maze}
        self.env = normalized_env.make_normalized_env(Minecraft(maze_def, reset=FLAGS.reset, grayscale=False,
                                                                vision_observation=FLAGS.vision,
                                                                video_dim=(FLAGS.height, FLAGS.width),
                                                                num_parallel=FLAGS.num_parallel)) # normalized_env.make_normalized_env(gym.make(FLAGS.env))
        tf.set_random_seed(FLAGS.tfseed)
        np.random.seed(FLAGS.npseed)
        #self.env.monitor.start(os.path.join(FLAGS.outdir, 'monitor'), force=FLAGS.force)
        #self.env.seed(FLAGS.gymseed)
        gym.logger.setLevel(gym.logging.WARNING)

        dimO = self.env.observation_space.shape
        dimA = self.env.action_space.shape
        print(dimO, dimA)
        #pprint.pprint(self.env.spec.__dict__)

        self.agent = Agent(dimO, dimA=dimA)
        simple_log_file = open(os.path.join(FLAGS.outdir, 'log.txt'), 'a')
        # Save command line arg
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        simple_log_file.write(" ".join(sys.argv[:] + [git_hash]))

        avg_rewards = []
        while self.train_timestep < FLAGS.total:

            # test
            reward_list = []
            for _ in xrange(FLAGS.test):
                reward, timestep = self.run_episode(test=True, monitor=np.random.rand() < FLAGS.monitor)
                reward_list.append(reward)
                self.test_timestep += timestep
            avg_reward = np.mean(reward_list)
            avg_rewards.append(avg_reward)
            print('Average test return {} after {} timestep of training.'.format(avg_reward, self.train_timestep))
            #print >> simple_log_file, "{}\t{}\t{}\t{}\t{}".format(self.train_timestep, avg_reward, np.std(reward_list), np.min(reward_list), np.max(reward_list))
            # Stopping criterion
            if self.train_timestep > 5e5 and len(avg_rewards) > 10 and np.var(avg_rewards) < 1:
                break


            # train
            train_reward_list = []
            last_checkpoint = self.train_timestep / FLAGS.train
            while self.train_timestep / FLAGS.train == last_checkpoint:
                reward, timestep = self.run_episode(test=False, monitor=False)
                train_reward_list.append(reward)
                self.train_timestep += timestep
            train_avg_reward = np.mean(train_reward_list)
            print('Average train return {} after {} timestep of training.'.format(train_avg_reward, self.train_timestep))
            simple_log_file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(self.train_timestep, avg_reward, np.std(reward_list), np.min(reward_list), np.max(reward_list), 
                                                                    train_avg_reward))
            simple_log_file.flush()

        #self.env.monitor.close()

    def run_episode(self, test=True, monitor=False):
        #self.env.monitor.configure(lambda _: monitor)
        observation = self.env.reset()
        self.agent.reset(observation)
        sum_reward = 0
        timestep = 0
        term = False
        while not term:
            action = self.agent.act(test=test)

            observation, reward, term, info = self.env.step(action)
            term = (not test and timestep + 1 >= FLAGS.tmax) or term

            filtered_reward = reward #self.env.filter_reward(reward)
            self.agent.observe(filtered_reward, term, observation, test=test)

            sum_reward += reward
            timestep += 1

        return sum_reward, timestep


def main():
    Experiment().run()

if __name__ == '__main__':
    #main()
    runtime_env.run(main, FLAGS.outdir)
