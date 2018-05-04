#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run

args = parse_args()

env = 'Walker2d'
#env = 'HalfCheetah'
#env = 'Hopper'
#task = 'Balancing'
task = 'Walking'


if task == 'Balancing':
    task_balancing = task
else:
    task_balancing = ''

args['cfg'] = "Roboschool{}-v1".format(env+task_balancing+'GRL')
#args['cfg'] = "Roboschool{}-v1".format(env+task_balancing)

args['steps'] = 0
args['trials'] = 100
args['test_interval'] = 0
args['normalize_observations'] = False
args['normalize_returns'] = False
args['batch_norm'] = True

#if env == 'Walker2d' and task == 'Walking':
#    args['load_file'] = 'cl/walker2d_walking-last'
#    args['compare_with'] = 'cl/walker2d_balancing-last'
#elif env == 'Hopper' and task == 'Walking':
#    args['load_file'] = 'cl/hopper_walking-last'
#    args['compare_with'] = 'cl/hopper_balancing-last'

args['load_file'] = 'ddpg-exp1_two_stage_walker2d_ga_w-g0001-mp0-02_walking-best'

args['output'] = ''
args['render'] = True
args['trajectory'] = 'cl/{}_{}'.format(env, task)
args['env_timestep'] = 0.0165


# Run actual script.
args['save'] = False
cfg_run(**args)
