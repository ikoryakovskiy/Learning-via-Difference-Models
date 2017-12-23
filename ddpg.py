#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from grlgym.envs.grl import GRLEnv
import argparse
import time
import yaml
import os
from ddpg_loop import start

from my_monitor import MyMonitor

def cfg_run(**config):
    with open("{}.yaml".format(config['output']), 'w', encoding='utf8') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    del config['cores']
    if config['seed'] == None:
        config['seed'] = int.from_bytes(os.urandom(4), byteorder='big')
    print(type(config['seed']))
    run(**config)

def run(cfg, **config):

    # Create envs.
    env = GRLEnv(cfg)
    env = MyMonitor(env, config['output'])
    env.seed(seed=config['seed'])

    start_time = time.time()
    start(env=env, **config)
    print('total runtime: {}s'.format(time.time() - start_time))

    env.close()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Flow options
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--tensorboard', type=bool, default=False)

    # Task execution
    parser.add_argument('--cfg', type=str, default='cfg/rbdl_py_balancing.yaml')
    parser.add_argument('--trials', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--train-steps', type=int, default=1)
    parser.add_argument('--test-interval', type=int, default=10)

    # Learning algorithm options
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--layer-norm', default=True)
    parser.add_argument('--normalize-returns', default=False)
    parser.add_argument('--normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--ou-sigma', type=float, default=0.15)
    parser.add_argument('--ou-theta', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)

    # Replay Buffer options
    parser.add_argument('--minibatch-size', type=int, default=64)
    parser.add_argument('--rb-max-size', type=int, default=300000)
    parser.add_argument('--rb-min-size', type=int, default=20000)
    parser.add_argument('--rb-save-filename', type=str, default='')
    parser.add_argument('--rb-load-filename', type=str, default='')

    # In/out options
    parser.add_argument('--output', type=str, default='default')
    parser.add_argument('--load-file', type=str, default='')
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()

    # Run actual script.
    cfg_run(**args)
