#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import yaml
import learning_params
from main_ddpg import start

def single_ddpg(cfg):
    learning = learning_params.init()

    params = dict(
        learning=learning,
        replay_buffer=dict(load=0, save=0, max_size=300000, min_size=20000, minibatch_size=64),
        transitions=dict(load=0, save=1, save_filename='db_trajectories', buffer_size=5000),
        difference_model=0
    )

    start(cfg, params)

if __name__ == "__main__":
    single_ddpg('../grl/qt-build/cfg/leo/drl/rbdl_ddpg.yaml')