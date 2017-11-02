#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import yaml
import global_params
from main_ddpg import start

global_params.init()
global_params.ou_sigma = 0
global_params.theta = 1
global_params.actor_learning_rate = 0
global_params.critic_learning_rate = 0

d = {'transitions': {'load': 0, 'load_filename': 'db_trajectories', 
                     'save': 0, 'save_filename': 'db_trajectories_play',
                     'buffer_size': 5000},
     'difference_model': 0}
     
with open('config.yaml', 'w') as yaml_file:
    yaml.dump(d, yaml_file, default_flow_style=False)

# Run the transitions on the original model
start('../grl/qt-build/cfg/leo/drl/rbdl_ddpg_play.yaml', global_params)

