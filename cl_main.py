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

def cl_run(tasks, cl_mode, **base_cfg):
    assert(base_cfg["trials"] == 0)
    assert(base_cfg["steps"]  != 0)
    assert(base_cfg['reach_reward'])

#    params = np.load(base_cfg['cl_load']+'.npy').squeeze()
#    rd = np.sum(np.square(params-1))
#    return (1*random.random() + rd, 'testing')

    ss = 0
    stage_counter = 0
    prev_config = None
    damage = 0
    env = None
    pt = PerformanceTracker(base_cfg["cl_input_norm"])

    cl_info = ''
    avg_test_return = base_cfg['reach_reward']

    while ss < base_cfg["steps"] and avg_test_return <= base_cfg['reach_reward']:
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
        env = Leo(config['cfg'])
        #env.report('all') # 'all' is used to get correct damage info
        env = MyMonitor(env, config['output'], report='all')

        # load previous stage actor, critic and curriculum
        if prev_config:
            config['cl_load'] = prev_config['cl_save']
            config['load_file'] = prev_config['output']
            config['rb_load_filename'] = prev_config['rb_save_filename']

        if cl_mode == 'walking':
            config['cl_on'] = False # forbid loading curriculum

        cl_info += cl_mode + ' '

        # run the stage
        avg_test_return, damage_new, ss_new, cl_mode = start(env=env, pt=pt, cl_mode=cl_mode, **config)

        damage += damage_new
        ss += ss_new
        prev_config = config.copy() # Dicts are mutable
        stage_counter += 1
        cl_info += ('{:d}'.format(ss_new)).ljust(7) + ' '

    if env:
        env.close()

    # calculate final performance
    walking_avg_damage = 4035.00

    print(base_cfg['output'] + ' finished!')
    if avg_test_return > base_cfg['reach_reward']:
        return (damage, cl_info)
    else:
        return (max([walking_avg_damage, damage]), cl_info)




