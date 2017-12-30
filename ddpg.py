#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from grlgym.envs.grl import Leo
import argparse
import time
import yaml
import os
from importlib import reload
from ddpg_loop import start
from my_monitor import MyMonitor

import gym
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import roboschool

def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser."""
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)

def cfg_run(**config):
    with open("{}.yaml".format(config['output']), 'w', encoding='utf8') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    del config['cores']
    if config['seed'] == None:
        config['seed'] = int.from_bytes(os.urandom(4), byteorder='big', signed=False) // 2
    run(**config)

def run(cfg, **config):
    # Create envs.
    if os.path.isfile(cfg):
        #from grlgym.envs.grl import Leo
        env = Leo(cfg)
    else:
        env = gym.make(cfg)

    env = MyMonitor(env, config['output'])

    start_time = time.time()
    start(env=env, **config)
    print('total runtime: {}s'.format(time.time() - start_time))

    env.close()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Flow options
    parser.add_argument('--cores', type=int, default=1)
    boolean_flag(parser,  'tensorboard', default=False)

    # Task execution
    parser.add_argument('--cfg', type=str, default='cfg/rbdl_py_balancing.yaml')
    parser.add_argument('--trials', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--test-interval', type=int, default=30)
    parser.add_argument('--curriculum', type=str, default='')
    boolean_flag(parser,  'render', default=False)

    # Learning algorithm options
    parser.add_argument('--tau', type=float, default=0.001)
    boolean_flag(parser,  'layer-norm', default=True)
    boolean_flag(parser,  'normalize-returns', default=False)
    boolean_flag(parser,  'normalize-observations', default=False)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--critic-l2-reg', type=float, default=0.001)
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
    parser.add_argument('--reassess-for', type=str, default='')

    # In/out options
    parser.add_argument('--output', type=str, default='default')
    parser.add_argument('--load-file', type=str, default='')
    boolean_flag(parser,  'save', default=False)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()

    # Run actual script.
    cfg_run(**args)
