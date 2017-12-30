#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run

args = parse_args()

#task = 'walking'
task = 'balancing'
env = 'hopper'

args['cfg'] = "RoboschoolHopper-v1"
args['steps'] = 50000
args['test_interval'] = 30
args['normalize_observations'] = False
args['normalize_returns'] = False
args['layer_norm'] = True
args['reassess_for'] = ''
args['output'] = 'hopper_{}'.format(task)

# Run actual script.
args['save'] = True
cfg_run(**args)
