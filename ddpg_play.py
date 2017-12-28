#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run

args = parse_args()

#task = 'walking'
task = 'balancing'

ld_option = 'best'

args['cfg'] = 'cfg/rbdl_py_{}_play.yaml'.format(task)
args['steps'] = 0
args['trials'] = 1
args['test_interval'] = 0
args['seed'] = 0
args['critic_l2_reg']= 0.001
args['tau']= 0.001
args['normalize_observations'] = False
args['normalize_returns'] = False
args['layer_norm'] = True
args['output'] = 'rbdl_py_{}_play'.format(task)
args['load_file'] = 'ddpg-balancing-5000000-1010-mp3-last' #'rbdl_py_{}-{}'.format(task, ld_option)

'''
import yaml
with open('tmp/ddpg-cfg_rbdl_py_balancing-10000000-000000-000000-000000-000000-000100-000000-mp0.yaml', 'r') as file:
    args = yaml.load(file)
'''

# Run actual script.
args['save'] = False
cfg_run(**args)
