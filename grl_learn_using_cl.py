#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddpg import parse_args
from cl_main import cl_run
from cl_learning import Helper
args = parse_args()

args['cl_on'] = True
args['rb_min_size'] = 1000
args['reach_return'] = 1422.66
args['default_damage'] = 4035.00
args['perf_td_error'] = True
args['perf_l2_reg'] = True
args['steps'] = 300000
args['cl_structure'] = '_1'
args['cl_depth'] = 1

# Parameters
starting_task = 'balancing'
tasks = {'balancing': 'cfg/leo_balancing.yaml', 'walking': 'cfg/leo_walking.yaml'}
hp = Helper(args, 'cl', 'ddpg', tasks, starting_task, 1, use_mp=False)

# Weights of the NN
sl_solution = '''0 0 1 -0.5'''

solution = [float(i) for i in sl_solution.split()]

# Run actual script.
config, tasks, starting_task = hp.gen_cfg([solution], 1)[0]
cl_run(tasks, starting_task, **config)
