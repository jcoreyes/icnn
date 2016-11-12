import os
from glob import glob
import numpy as np


def get_data(log_file):
    with open(log_file) as f:
        lines = [line.split() for line in f.readlines()[2:]]
        data = np.asfarray(lines)

        if len(data) == 0:
            return [None] * 6
        steps, avg_r, std_r, min_r, max_r, train_r = [data[:, i] for i in range(6)]
    return steps, avg_r, std_r, min_r, max_r, train_r

PATH = 'ddpgtmazenewrvision'
log_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.txt'))]

perf = []
thresh = - 300
done = 0
for log_file in log_files:
    steps, avg_r, std_r, min_r, max_r, train_r = get_data(log_file)
    if steps is not None:
        done += 1
        print steps[-1], avg_r[-1], train_r[-1]
        if avg_r[-1] > thresh:
            min_steps = steps[avg_r<thresh].max()
            perf.append((min_steps, log_file))

for x in sorted(perf):
    print x
print len(log_files)
print done
print len(perf)