#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run

args = parse_args()

env = 'hopper'
#task = 'walking'
task = 'balancing'


args['cfg'] = "RoboschoolHopper-v1"
args['steps'] = 300000
args['test_interval'] = 30
args['normalize_observations'] = False
args['normalize_returns'] = False
args['layer_norm'] = True
args['output'] = '{}_{}'.format(env, task)

# Run actual script.
args['save'] = True
cfg_run(**args)
