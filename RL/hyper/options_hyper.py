import numpy as np
from itertools import product
from subprocess import call
import os
import random
from multiprocessing import Pool


hyper_param = {
'batchnorm': (True, False),
'rate': (0.001, 0.0001),
'prate': (0.001, 0.0001, 0.00001),
'outheta':(0.0, 0.15, 0.30),
#'ousigma':(0.0, 0.1, 0.2),
'reward_k':(10, 1.0, 0.1, 0.01),
'stochastic_options':(True, False)
}


param_names = hyper_param.keys()
grid = set(product(*[set(x) for x in hyper_param.values()]))
MAIN_FILE = "../src/main_minecraft.py"
OUTDIR = 'ddpgoptionstmazevision/'
NUM_PARALLEL = 3

default_args = ['--model', 'DDPGOptions', '--vision', 'True', '--width', str(32), '--height', str(32),
                '--force', 'True', '--maze', 'TMaze', '--num_parallel', str(NUM_PARALLEL)]

def create_command(param_names, params):
    command_args = []
    path = OUTDIR
    for param_name, param in zip(param_names, params):
        command_args.append("--" + param_name)
        command_args.append(str(param))
        path = path + "%s=%s/" %(param_name, param)
    assert os.path.isdir(path)
    command_args.extend(['--outdir', path])
    command_args.extend(default_args)
    command = ["python", MAIN_FILE] + command_args
    return command 

def create_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def create_nested_dir(param_names, hyper_param):
    # Create tree dir with all hyper param values as dir name
    prepends = [OUTDIR]
    for param_name in param_names:
        new_prepends = []
        for prepend in prepends:
            for param_value in hyper_param[param_name]:
                path = prepend + "%s=%s" %(param_name, param_value)
                create_dir(path)
                new_prepends.append(path + '/')
        prepends = new_prepends[:]

# Create output dirs for each hyperparam config
create_nested_dir(param_names, hyper_param)
pool = Pool(NUM_PARALLEL)
command_lst = [create_command(param_names, params) for params in grid]
result = pool.map_async(call, command_lst)
#print result.get()
pool.close()
pool.join()
# for params in grid:
#     command = create_command(param_names, params)
#     print command
#     call(command)
#     break
