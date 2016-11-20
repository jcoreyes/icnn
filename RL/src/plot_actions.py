import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm


def plot_episode(action_data):
    num_options = action_data.shape[1] - 2
    step_no, x, z = [action_data[:, i] for i in range(3)]
    option_idx = action_data[:, -num_options:].argmax(axis=1) 
    colors = cm.rainbow(np.linspace(0, 1, num_options))
    plt.figure()
    for i in range(num_options):
        idx = option_idx == i
        plt.scatter(x[idx], z[idx], color=colors[i]) 
    plt.savefig('action_plot.png')
action_data_file = sys.argv[1]
lines = open(action_data_file).readlines()


for line in lines:
    # Organized as train_timestep s1 s2 s3
    # s1 = step_no,x,z,o1,o2,o3
    line_data  = line.rstrip().split('\t')
    train_timestep = line_data[0]
    action_data = [x.split(',') for x in line_data[1:-1]]
    
    action_data = np.asarray(action_data)
    plot_episode(action_data)
    break

 
