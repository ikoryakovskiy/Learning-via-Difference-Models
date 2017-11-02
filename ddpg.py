#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import yaml
import global_params
from main_ddpg import start

def single_ddpg(cfg):
    global_params.init()

    dcfg = {'transitions': dict(load=0, save=1, save_filename='db_trajectories', buffer_size=5000),
            'difference_model': 0}

    with open(cfg, 'r') as f:
        config = yaml.load(f)
        output = config["experiment"]["output"]

    ddpg_cfg = "py_" + output + ".yaml"
    with open(ddpg_cfg, 'w') as yaml_file:
        yaml.dump(dcfg, yaml_file, default_flow_style=False)

    start(cfg, global_params)

if __name__ == "__main__":
    single_ddpg('../grl/qt-build/cfg/leo/drl/rbdl_ddpg.yaml')