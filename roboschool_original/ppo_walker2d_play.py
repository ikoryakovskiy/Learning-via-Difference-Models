#!/usr/bin/env python3

from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
import os

import roboschool, gym

def train(env_id, num_timesteps, seed, save_file, load_file, render, stochastic):
    from baselines.ppo1 import mlp_policy
    import my_pposgd_simple as pposgd_simple
    sess = U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
            sess=sess, save_file=save_file, load_file=load_file, render=render, stochastic=stochastic
        )
    env.close()

def main():
    args = mujoco_arg_parser().parse_args()
    args.env = 'RoboschoolWalker2d-v1'
    args.save_file = ''
    args.load_file = 'ppo_walker2d'
    args.num_timesteps = 10000000
    args.render = True
    args.stochastic = False
    print(args)
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          save_file=args.save_file, load_file=args.load_file,
          render=args.render, stochastic=args.stochastic)

if __name__ == '__main__':
    main()
