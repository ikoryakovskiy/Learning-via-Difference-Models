#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GRL should be imported before tensorflow.
# Otherwise, error : "dlopen: cannot load any more object with static TLS"
try:
    from grlgym.envs.grl import Leo
except ImportError:
    pass

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ddpg_loop import start
from my_monitor import MyMonitor
from ptracker import PerformanceTracker
import random
import numpy as np
from os.path import exists
import yaml, io
import collections
import pdb

import importlib
if importlib.util.find_spec("roboschool"):
    import gym, roboschool
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def env_connect(path):
    if os.path.isfile(path):
        env = Leo(path)
    else:
        env = gym.make(path)
    return env


def cl_run(tasks, cl_mode, **base_cfg):
    assert(base_cfg["trials"] == 0)
    assert(base_cfg["steps"]  != 0)

    if (isinstance(base_cfg["steps"], list) or isinstance(base_cfg["steps"], tuple)) and len(base_cfg['steps']) >= len(tasks.keys()):
        steps = sum(base_cfg["steps"])
        step_based_cl_switching = True
    else:
        steps = base_cfg["steps"]
        step_based_cl_switching = False

    if isinstance(base_cfg['reach_timeout'], collections.Sequence) and len(base_cfg['reach_timeout']) >= len(tasks.keys()):
        reach_timeout_based_cl_switching = True
    else:
        reach_timeout_based_cl_switching = False
#    ################
#    if step_based_cl_switching:
#        ss = base_cfg["steps"]
#        damage = 0.0001 * (abs(ss[0]-20000) + abs(ss[1]-30000))
#        if random.random() > 0.:
#            return (9*random.random() + damage, 'testing', [])
#        else:
#            1/0
#            return (None, None, None)
#    else:
#        params = np.load(base_cfg['cl_load']+'.npy').squeeze()
#        reg = base_cfg['cl_l2_reg'] * np.linalg.norm(params, ord=2)
#        if random.random() > 0.:
#            return (1000*random.random() + reg, 'testing', [])
#        else:
#            #1/0
#            return (None, None, None)
#    ################

    print('cl_run: ' +  base_cfg['output'] + ' started!')

    ss = 0
    stage_counter = 0
    prev_config = None
    damage = 0
    env = None
    pt = PerformanceTracker(base_cfg)
    if base_cfg["cl_pt_load"]:
        pt.load(base_cfg["cl_pt_load"])

    cl_info = ''
    avg_test_return = base_cfg['reach_return']
    task_sequence = ('balancing_tf', 'balancing', 'walking')

    while ss < steps and (not base_cfg['reach_return'] or avg_test_return <= base_cfg['reach_return']):
        stage = '-{:02d}_'.format(stage_counter) + cl_mode
        config = base_cfg.copy() # Dicts are mutable

        config['cfg'] = tasks[cl_mode]
        config['output']  = base_cfg['output']  + stage
        config['save']    = base_cfg['output']  + stage
        config['rb_save_filename'] = base_cfg['output']  + stage
        if config['seed'] == None:
            config['seed'] = int.from_bytes(os.urandom(4), byteorder='big', signed=False) // 2
        if base_cfg['cl_save']:
            config['cl_save'] = base_cfg['cl_save'] + stage
        if base_cfg['trajectory']:
            filename, file_extension = os.path.splitext(base_cfg['trajectory'])
            config['trajectory'] = filename + stage + file_extension
        if step_based_cl_switching:
            config['steps'] = int(base_cfg["steps"][stage_counter])
        else:
            config['steps']   = steps - ss
        if reach_timeout_based_cl_switching:
            config['reach_timeout'] = base_cfg['reach_timeout'][stage_counter]

        if not base_cfg['cl_keep_samples']:
            config['rb_max_size'] = config['steps']

        # every stage happens when environment is switched over, thus we initialise it every stage
        if env:
            env.close()
            env = None
        #pdb.set_trace()
        env = env_connect(config['cfg'])
        env = MyMonitor(env, config['output'], report='all')
        if base_cfg['measurment_noise']:
            env.reconfigure({'measurment_noise': base_cfg['measurment_noise']})
        if base_cfg['actuation_noise']:
            env.reconfigure({'actuation_noise': base_cfg['actuation_noise']})

        # load previous stage actor, critic and curriculum
        if prev_config:
            config['cl_load'] = prev_config['cl_save']
            if not base_cfg['options']:
                config['load_file'] = prev_config['output']
                config['rb_load_filename'] = prev_config['rb_save_filename']
            else:
                opt = base_cfg['options'][cl_mode]
                if 'nnload' in opt:
                    config['load_file'] = prev_config['output']
                if 'rbload_re' in opt:
                    config['rb_load_filename'] = prev_config['rb_save_filename']
                    config['reassess_for'] = opt.split('rbload_re_')[1]
                elif 'rbload' in opt:
                    config['rb_load_filename'] = prev_config['rb_save_filename']

        if cl_mode == 'walking':
            config['cl_structure'] = '' # forbid loading curriculum
            config['rb_save_filename'] = '' # do not save replay beffer since it will not be used anyway

        cl_info += cl_mode + ' '

        # DBG: export configuration
        with io.open(config['output']+'.yaml', 'w', encoding='utf8') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

        # run the stage
        avg_test_return, damage_new, ss_new, cl_mode_new = start(env=env, pt=pt, cl_mode=cl_mode, **config)

        damage += damage_new
        ss += ss_new
        prev_config = config.copy() # Dicts are mutable
        stage_counter += 1
        cl_info += ('{:d}'.format(ss_new)).ljust(7) + ' '

        print('cl_run: {} stage {} done'.format(config['output'], stage))
        if cl_mode == 'walking':
            print('cl_run: {} exit from the loop {}'.format(config['output'], ss < steps))
            break

        if step_based_cl_switching or reach_timeout_based_cl_switching:
            idx = [idx for idx, ts in enumerate(task_sequence) if ts == cl_mode][0]
            cl_mode_new = task_sequence[idx+1]
        cl_mode = cl_mode_new

    if env:
        env.close()

    # notify
    print('cl_run: ' +  base_cfg['output'] + ' finished!')

    # calculate final performance, if default damage is provided
    if base_cfg['reach_return'] and base_cfg['default_damage']:
        walking_avg_damage = base_cfg['default_damage']

        # penalize if target performance was not reached
        if avg_test_return < base_cfg['reach_return']:
            damage = max([walking_avg_damage, damage])

        # penalize absence of walking stage
        if "walking" not in cl_info:
            damage = 2*walking_avg_damage

    # add regularization
    reg = 0
    params = []
    if exists(base_cfg["cl_load"]+'.npy'):
        params = np.load(base_cfg["cl_load"]+'.npy').squeeze()
        if base_cfg['cl_l2_reg']:
            reg = base_cfg['cl_l2_reg'] * np.square(1-np.linalg.norm(params, ord=2))

    # return final performance
    return (damage + reg, cl_info, list(params))
