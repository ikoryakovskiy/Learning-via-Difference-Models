#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run

args = parse_args()

#task = 'walking'
task = 'balancing'

args['cfg'] = 'cfg/rbdl_py_{}.yaml'.format(task)
args['steps'] = 50000
args['test_interval'] = 30
#args['seed'] = 1
#args['curriculum'] = 'rwForward_50_300_10'
args['normalize_observations'] = False
args['normalize_returns'] = False
args['layer_norm'] = True
args['reassess_for'] = '{}'.format(task)
args['output'] = 'rbdl_py_{}'.format(task)

#args['rb_save_filename'] = 'rbdl_py_{}'.format(task)
#args['rb_load_filename'] = 'rbdl_py_balancing'

#args['rb_load_filename'] = 'rbdl_py_{}'.format(task)
#args['load_file'] = 'rbdl_py_balancing'
#args['tensorboard'] = True

'''
import yaml
with open('tmp/ddpg-walking-30000000-30000000--150000--200000-1000-mp0.yaml', 'r') as file:
#with open('tmp/ddpg-balancing-5000000-1010-mp0.yaml', 'r') as file:
    args = yaml.load(file)
args['seed'] = 1
#args['steps'] = 500
'''
# Run actual script.
args['save'] = True
cfg_run(**args)
