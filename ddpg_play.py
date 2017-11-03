#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import yaml
import learning_params
from main_ddpg import start

def single_ddpg_play(cfg):
    learning = learning_params.init()
    learning["ou_sigma"] = 0
    learning["theta"] = 1
    learning["actor_learning_rate"] = 0
    learning["critic_learning_rate"] = 0

    params = dict(
        learning=learning,
        transitions=dict(load=0, save=1, save_filename='db_trajectories', buffer_size=5000),
        difference_model=0
    )

    start(cfg, params)

if __name__ == "__main__":
    single_ddpg_play('../grl/qt-build/cfg/leo/drl/rbdl_ddpg_play.yaml')


