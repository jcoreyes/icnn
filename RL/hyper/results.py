import os
from glob import glob
import numpy as np
import sys
import matplotlib.pyplot as plt

PATH = sys.argv[1]
THRESH = 500
log_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.txt'))]


def get_data(log_file):
    with open(log_file) as f:
        lines = [line.split() for line in f.readlines()[2:]]
        data = np.asfarray(lines)

        if len(data) == 0:
            return [None] * 6
        steps, avg_r, std_r, min_r, max_r, train_r = [data[:, i] for i in range(6)]
    return steps, avg_r, std_r, min_r, max_r, train_r

def plot_data(data, split):
    num_plots = len(data)
    plt.figure(figsize=(20, 10))
    colormap = plt.cm.gist_ncar
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

    for x in sorted(data):
        min_steps, log_file, reward = x
        print min_steps, log_file
        plt.plot(steps, reward, label=','.join(log_file.split('/')[1:-1]))
    print len(log_files)

    print len(data)
    plt.legend(loc='lower right')
    plt.title(split + ' reward')
    plt.savefig(PATH + split + '_results.png')
    #plt.show()

train_perf = []
test_perf = []
done = 0

for log_file in log_files:
    steps, test_r, std_r, min_r, max_r, train_r = get_data(log_file)
    if steps is not None:

        done += 1
        print steps[-1], test_r[-1], train_r[-1]
        if train_r[-1] > THRESH:
            min_steps = steps[train_r<THRESH].max()
            train_perf.append((min_steps, log_file, train_r))
        if test_r[-1] > THRESH:
            min_steps = steps[test_r<THRESH].max()
            if min_steps < 20000:
                test_perf.append((min_steps, log_file, test_r))


plot_data(train_perf, 'train')
plot_data(test_perf, 'test')
print "Finished %d runs" %done
