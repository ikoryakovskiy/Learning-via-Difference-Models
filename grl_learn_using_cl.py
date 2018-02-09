#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddpg import parse_args
from cl_main import cl_run
from cl_learning import Helper
args = parse_args()

args['cl_on'] = True
args['rb_min_size'] = 1000
args['reach_return'] = 1422.66
args['steps'] = 300000
args['cl_structure'] = '_1'
args['cl_depth'] = 2

# Parameters
starting_task = 'balancing'
tasks = {'balancing': 'cfg/leo_balancing.yaml', 'walking': 'cfg/leo_walking.yaml'}
hp = Helper(args, 'cl', 'ddpg', tasks, starting_task, 1, use_mp=False)

# Weights of the NN
sl_solution = '''0.18473815  1.60566807 -2.5843519   0.42126614  0.83710431 -1.36806102
 -0.31722035'''

solution = [float(i) for i in sl_solution.split()]

# Run actual script.
config, tasks, starting_task = hp.gen_cfg([solution], 1)[0]
cl_run(tasks, starting_task, **config)
