#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run
'''
args = parse_args()

task = 'walking'
#task = 'balancing'

args['cfg'] = 'cfg/rbdl_py_{}.yaml'.format(task)
args['task_name'] = '{}'.format(task)
args['steps'] = 50000
args['test_interval'] = 30
args['seed'] = 1
args['normalize_observations'] = False
args['normalize_returns'] = False
args['layer_norm'] = True
args['re_evaluate'] = True
args['output'] = 'rbdl_py_{}'.format(task)
#args['rb_save_filename'] = 'rbdl_py_{}'.format(task)
#args['rb_load_filename'] = 'rbdl_py_{}'.format(task)
args['rb_load_filename'] = 'rbdl_py_balancing'
args['load_file'] = 'rbdl_py_balancing'

#args['curriculum'] = 'rwForward_50_300_10'

#args['tensorboard'] = True

'''
import yaml
with open('tmp/ddpg-walking_after_balancing-25000000-000000-1001-mp1.yaml', 'r') as file:
    args = yaml.load(file)


# Run actual script.
#args['save'] = True
cfg_run(**args)
