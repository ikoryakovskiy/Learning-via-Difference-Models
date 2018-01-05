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
args['steps'] = 0
args['trials'] = 10
args['test_interval'] = 0
args['normalize_observations'] = False
args['normalize_returns'] = False
args['layer_norm'] = True
#args['load_file'] = '{}_{}'.format(env.lower(), task.lower())
args['load_file'] = 'ddpg-Walker2d_walking_after_balancing-70000000-1101-mp1-best'.format(env.lower(), task.lower())
args['output'] = '{}_{}_play'.format(env.lower(), task.lower())
args['render'] = True

# Run actual script.
args['save'] = False
cfg_run(**args)
