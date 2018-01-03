#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run

args = parse_args()

env = 'leo'
#task = 'walking'
task = 'balancing'

args['cfg'] = 'cfg/{}_{}_play.yaml'.format(env, task)
args['steps'] = 0
args['trials'] = 10
args['test_interval'] = 0
args['normalize_observations'] = False
args['normalize_returns'] = False
args['layer_norm'] = True
args['output'] = '{}_{}_play'.format(env, task)
args['load_file'] = '{}_{}'.format(env, task)

# Run actual script.
args['save'] = False
cfg_run(**args)
