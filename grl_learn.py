#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run
'''
args = parse_args()

env = 'leo'
task = 'walking'
#task = 'balancing'

args['cfg'] = 'cfg/{}_{}_export.yaml'.format(env, task)
args['steps'] = 2000
args['test_interval'] = 30
args['seed'] = 1
args['normalize_observations'] = False
args['normalize_returns'] = False
args['batch_norm'] = True
#args['reassess_for'] = 'walking_300_-1.5'
args['output'] = '{}_{}'.format(env, task)

#args['rb_save_filename'] = 'rbdl_py_{}'.format(task)
#args['rb_load_filename'] = 'rbdl_py_balancing'

#args['rb_load_filename'] = 'rbdl_py_{}'.format(task)
#args['load_file'] = 'rbdl_py_balancing'
#args['tensorboard'] = True

'''
import yaml
with open('tmp/ddpg-balancing-5000000-6400000-1010-mp0.yaml', 'r') as file:
    args = yaml.load(file)
args['seed'] = 1
#args['steps'] = 500


# Run actual script.
args['save'] = True
cfg_run(**args)
