#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run

args = parse_args()

env = 'Hopper'
#env = 'Walker2d'
#env = 'HalfCheetah'
task = 'Balancing'
#task = 'Walking'

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
args['load_file'] = 'ddpg-Hopper_balancing-20000000-1010-mp0'
args['output'] = '{}_{}_play'.format(env.lower(), task.lower())
args['render'] = True

# Run actual script.
args['save'] = False
cfg_run(**args)
