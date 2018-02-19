#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run

args = parse_args()

env = 'leo'
task = 'walking'
#task = 'balancing'
#task = 'crouching'

args['cfg'] = 'cfg/{}_{}_play.yaml'.format(env, task)
args['steps'] = 0
args['trials'] = 1
args['test_interval'] = 0
args['normalize_observations'] = False
args['normalize_returns'] = False
args['batch_norm'] = True
args['output'] = '{}_{}_play'.format(env, task)
#args['load_file'] = '{}_{}'.format(env, task)
args['load_file'] = 'ddpg-walking_after_balancing-25000000-100000000-1100-mp0'

# Run actual script.
args['save'] = False
cfg_run(**args)
