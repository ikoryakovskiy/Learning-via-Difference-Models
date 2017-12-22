#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run

args = parse_args()

task = 'walking'
#task = 'balancing'

args['cfg'] = 'cfg/rbdl_py_{}.yaml'.format(task)
args['steps'] = 300000
args['test_interval'] = 30
args['critic_l2_reg']= 0.001
args['tau']= 0.001
args['normalize_observations'] = False
args['normalize_returns'] = False
args['layer_norm'] = True
args['output'] = 'rbdl_py_{}'.format(task)


'''
import yaml
with open('tmp/ddpg-cfg_rbdl_py_balancing-10000000-000000-000000-000000-000000-000100-000000-mp0.yaml', 'r') as file:
    args = yaml.load(file)
'''

# Run actual script.
args['save'] = True
cfg_run(**args)
