#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run

args = parse_args()

env = 'Walker2d'
#env = 'HalfCheetah'
#task = 'Balancing'
task = 'Walking'

if task == 'Balancing':
    task_balancing = task
else:
    task_balancing = ''

args['cfg'] = "Roboschool{}-v1".format(env+task_balancing)
args['steps'] = 1000000
args['test_interval'] = 30
#args['ou_sigma'] = 0.3
args['normalize_observations'] = False
args['normalize_returns'] = False
args['layer_norm'] = True
args['output'] = '{}_{}'.format(env.lower(), task.lower())

# Run actual script.
args['save'] = True
cfg_run(**args)
