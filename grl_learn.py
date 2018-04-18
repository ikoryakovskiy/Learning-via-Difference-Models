#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run

args = parse_args()

env = 'leo'
task = 'walking'
#task = 'balancing'

args['cfg'] = 'cfg/{}_{}.yaml'.format(env, task)
args['steps'] = 10
args['test_interval'] = 30
args['seed'] = 1
args['perf_td_error'] = True
args['perf_l2_reg'] = True
args['rb_min_size'] = 1000
args['normalize_observations'] = False
args['normalize_returns'] = False
args['batch_norm'] = True
#args['reassess_for'] = 'walking_300_-1.5'
args['output'] = 'cl/{}_{}'.format(env, task)

args['rb_save_filename'] = 'cl/{}_{}'.format(env, task)
#args['rb_load_filename'] = 'rbdl_py_balancing'


'''
import yaml
with open('tmp/ddpg-balancing-5000000-6400000-1010-mp0.yaml', 'r') as file:
    args = yaml.load(file)
args['seed'] = 1
#args['steps'] = 500
'''

# Run actual script.
args['save'] = True
cfg_run(**args)
