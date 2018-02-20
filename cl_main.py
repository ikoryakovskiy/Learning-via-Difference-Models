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

def cl_run(tasks, cl_mode, **base_cfg):
    assert(base_cfg["trials"] == 0)
    assert(base_cfg["steps"]  != 0)
    assert(base_cfg['reach_return'])


#    params = np.load(base_cfg['cl_load']+'.npy').squeeze()
#    reg = base_cfg['cl_l2_reg'] * np.linalg.norm(params, ord=2)
#    return (0*random.random() + reg, 'testing')


    print('cl_run: ' +  base_cfg['output'] + ' started!')
    ss = 0
    stage_counter = 0
    prev_config = None
    damage = 0
    env = None
    pt = PerformanceTracker(depth=base_cfg['cl_depth'], input_norm=base_cfg["cl_input_norm"])

    cl_info = ''
    avg_test_return = base_cfg['reach_return']

    while ss < base_cfg["steps"] and avg_test_return <= base_cfg['reach_return']:
        stage = '-{:02d}_'.format(stage_counter) + cl_mode
        config = base_cfg.copy() # Dicts are mutable

        config['cfg'] = tasks[cl_mode]
        config['steps']   = base_cfg["steps"] - ss
        config['output']  = base_cfg['output']  + stage
        config['save']    = base_cfg['output']  + stage
        config['cl_save'] = base_cfg['cl_save'] + stage
        config['rb_save_filename'] = base_cfg['output']  + stage
        if config['seed'] == None:
            config['seed'] = int.from_bytes(os.urandom(4), byteorder='big', signed=False) // 2

        # every stage happens when environment is switched over, thus we initialise it every stage
        if env:
            env.close()
            env = None
        env = Leo(config['cfg'])
        env = MyMonitor(env, config['output'], report='all')

        # load previous stage actor, critic and curriculum
        if prev_config:
            config['cl_load'] = prev_config['cl_save']
            config['load_file'] = prev_config['output']
            config['rb_load_filename'] = prev_config['rb_save_filename']

        if cl_mode == 'walking':
            config['cl_on'] = 0 # forbid loading curriculum

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

        # DBG: ensure exit from the loop after the walking stage
        print('cl_run: {} stage {} done'.format(config['output'], stage))
        if cl_mode == 'walking':
            print('cl_run: {} exit from the loop {} {}'.format(config['output'], ss < base_cfg["steps"], avg_test_return <= base_cfg['reach_return']))
            break
        # DBG end

        cl_mode = cl_mode_new

    if env:
        env.close()

    # calculate final performance
    walking_avg_damage = base_cfg['default_damage']

    # add solution regularization
    reg = 0
    params = []
    if exists(base_cfg["cl_load"]+'.npy'):
        params = np.load(base_cfg["cl_load"]+'.npy').squeeze()
        if base_cfg['cl_l2_reg']:
            reg = base_cfg['cl_l2_reg'] * np.abs(1-np.linalg.norm(params, ord=2))

    print('cl_run: ' +  base_cfg['output'] + ' finished!')
    if avg_test_return > base_cfg['reach_return']:
        return (damage + reg, cl_info, list(params))
    else:
        return (max([walking_avg_damage, damage]) + reg, cl_info, list(params))




