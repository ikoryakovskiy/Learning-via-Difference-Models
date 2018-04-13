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
args['load_file'] = 'cl/leo_walking-last'
args['compare_with'] = 'cl/leo_walking-last'
#args['compare_with'] = 'cl/leo_balancing-last'

args['trajectory'] = 'cl/{}_{}'.format(env, task)
args['env_timestep'] = 0.03
#args['env_report'] = 'all'

# Run actual script.
args['save'] = False
cfg_run(**args)
