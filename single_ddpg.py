#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import yaml
import global_params
from main_ddpg import start

global_params.init()

#global_params.ou_sigma = 0.1
#global_params.ou_theta = 0.15
d = {'transitions': {'load': 0, 'save': 1, 'save_filename': 'db_trajectories', 'buffer_size': 5000},
     'difference_model': 0}
with open('config.yaml', 'w') as yaml_file:
    yaml.dump(d, yaml_file, default_flow_style=False)

start('../grl/qt-build/cfg/leo/drl/rbdl_ddpg.yaml')

