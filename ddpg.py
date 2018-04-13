#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GRL should be imported before tensorflow.
# Otherwise, error : "dlopen: cannot load any more object with static TLS"
try:
    from grlgym.envs.grl import Leo
except ImportError:
    pass

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
        env = Leo(cfg)
    else:
        import roboschool
        env = gym.make(cfg)

    env = MyMonitor(env, config['output'], report=config['env_report'])

    start_time = time.time()
    start(env=env, **config)
    print('total runtime: {}s'.format(time.time() - start_time))

    env.close()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Flow options
    parser.add_argument('--cores', type=int, default=1)
    boolean_flag(parser,  'tensorboard', default=False)
    parser.add_argument('--version', type=int, default=0)
    boolean_flag(parser,  'mp_debug', default=False)
    parser.add_argument('--options', type=dict, default=None, help='Options which specify what to reload at each curriculum stage')

    # Task execution
    parser.add_argument('--cfg', type=str, default='', help='GRL yaml configuration or GYM model name') #cfg/leo_walking.yaml
    parser.add_argument('--env-timeout', type=float, default=20.0)
    parser.add_argument('--env-timestep', type=float, default=0.03)
    parser.add_argument('--env-td-error-scale', type=float, default=600.0, help='Approximate scale of TD errors')
    parser.add_argument('--env-report', type=str, default='test')
    parser.add_argument('--trials', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000) #
    parser.add_argument('--reach_timeout', type=float, default=0, help='Finish if trial happend to be longer then reach_balance reach_timeout_num in a row. 0 means desabled')
    parser.add_argument('--reach_timeout_num', type=float, default=0, help='Number of times reach_timeout should be reached in a row. 0 means desabled')
    parser.add_argument('--reach-return', type=float, default=None)
    parser.add_argument('--default-damage', type=float, default=None)
    parser.add_argument('--test-interval', type=int, default=30)
    parser.add_argument('--curriculum', type=str, default='')
    boolean_flag(parser,  'render', default=False)

    # Curriculum
    parser.add_argument('--cl-structure', type=str, default='') # cl:relu_3;tanh_1
    parser.add_argument('--cl-stages', type=str, default='') # balancing_tf;balancing;walking:monotonic
    parser.add_argument('--cl-load', type=str, default='')
    parser.add_argument('--cl-l2-reg', type=float, default=0.001)
    parser.add_argument('--cl-tau', type=float, default=0.001)
    parser.add_argument('--cl-lr', type=float, default=0.001)
    parser.add_argument('--cl-dropout-keep', type=float, default=1.0)
    parser.add_argument('--cl-cmaes-sigma0', type=float, default=4.0)
    boolean_flag(parser,  'cl-batch-norm', default=False)
    boolean_flag(parser,  'cl-input-norm', default=False)
    boolean_flag(parser,  'cl-running-norm', default=False)
    parser.add_argument('--cl-depth', type=int, default=1)
    parser.add_argument('--cl-reparam', type=str, default='spherical')
    boolean_flag(parser,  'cl-target', default=False)
    parser.add_argument('--cl-save', type=str, default='')
    parser.add_argument('--cl-pt-load', type=str, default='')
    parser.add_argument('--cl-pt-shape', type=tuple, default=None)
    boolean_flag(parser,  'cl-keep_samples', default=True)

    # Comparison of tasks (e.g. walking and balancing)
    parser.add_argument('--compare-with', type=str, default='')

    # Learning algorithm options
    parser.add_argument('--tau', type=float, default=0.001)
    boolean_flag(parser,  'batch-norm', default=True)
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
    parser.add_argument('--rb-min-size', type=int, default=1000)
    parser.add_argument('--rb-save-filename', type=str, default='')
    parser.add_argument('--rb-load-filename', type=str, default='')
    parser.add_argument('--reassess-for', type=str, default='')

    # Performance trackers
    boolean_flag(parser,  'perf-td-error', default=False)
    boolean_flag(parser,  'perf-l2-reg', default=False)
    boolean_flag(parser,  'perf-action-grad', default=False)
    boolean_flag(parser,  'perf-actor-grad', default=False)

    # In/out options
    parser.add_argument('--output', type=str, default='default')
    parser.add_argument('--trajectory', type=str, default=None)
    parser.add_argument('--load-file', type=str, default='')
    boolean_flag(parser,  'save', default=False)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()

    # Run actual script.
    cfg_run(**args)
