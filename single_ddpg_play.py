#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import yaml
import global_params
from main_ddpg import start

global_params.init()

d = {'transitions': {'load': 1, 'load_filename': 'data_train_ddpg', 'save': 1,
                       'save_filename': 'data_train_ddpg_play', 'buffer_size': 5000},
     'difference_model': 0}
with open('config.yaml', 'w') as yaml_file:
    yaml.dump(d, yaml_file, default_flow_style=False)

# Run the transitions on the original model
start('../grl/qt-build/cfg/leo/drl/rbdl_ddpg_play.yaml')

