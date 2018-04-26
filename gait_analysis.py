#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import parse_args, cfg_run
import os

stage_names = ('00_balancing_tf', '01_balancing', '02_walking')

def main():
    folder = 'gait_analysis/'
    policies = folder + 'policies/'
    trajectories = folder + 'trajectories/'
    misc = folder + 'misc/'

#    for mp in range(6):
#        leo_export(mp, task='balancing', policies=policies, trajectories=trajectories, misc=misc)
#        leo_export(mp, task='walking', policies=policies, trajectories=trajectories, misc=misc)

#    mujoco_export('Walker2d', 1, task='Balancing', policies=policies, trajectories=trajectories, misc=misc)
#    mujoco_export('Hopper', 2, task='Walking', policies=policies, trajectories=trajectories, misc=misc)
#    mujoco_export('HalfCheetah', 0, task='Balancing', policies=policies, trajectories=trajectories, misc=misc)
#    mujoco_export('HalfCheetah', 0, task='Walking', policies=policies, trajectories=trajectories, misc=misc)

    mujoco_models = ['Hopper', 'HalfCheetah', 'Walker2d']
    for env in mujoco_models:
        for mp in range(6):
            #mujoco_export(env, mp, task='Balancing', policies=policies, trajectories=trajectories, misc=misc)
            mujoco_export(env, mp, task='Walking', policies=policies, trajectories=trajectories, misc=misc)


def mujoco_export(env, mp, task='Walking', policies='', trajectories='', misc=''):
    args = parse_args()

    if task == 'Balancing':
        task_balancing = task
    else:
        task_balancing = ''

    args['cfg'] = "Roboschool{}-v1".format(env+task_balancing+'GRL')
    args['steps'] = 0
    args['trials'] = 1
    args['test_interval'] = 0
    args['normalize_observations'] = False
    args['normalize_returns'] = False
    args['batch_norm'] = True
    args['render'] = True

    args['output'] = misc + '{}_{}_play-mp{}'.format(env.lower(), task.lower(), mp)
    t = task[0].lower()
    if t == 'b':
        stage = stage_names[1]
    elif t == 'w':
        stage = stage_names[2]
    else:
        raise ValueError('incorrect task ' + task)
    args['load_file'] = policies + 'ddpg-exp1_two_stage_{env}_ga_{task}-g0001-mp{mp}-{stage}'.format(
            env=env.lower(), task=t, mp=mp, stage=stage)
    args['compare_with'] = policies + 'ddpg-exp1_two_stage_{env}_ga_b-g0001-mp{mp}-01_balancing-last'.format(
            env=env.lower(), mp=mp)
    args['trajectory'] = trajectories + '{}_{}-mp{}'.format(env.lower(), task.lower(), mp)

    args['env_timestep'] = 0.0165

    # Run actual script.
    args['save'] = False
    cfg_run(**args)


def leo_export(mp, policies='', task='walking', trajectories='', misc=''):
    args = parse_args()

    env = 'leo'

    args['cfg'] = 'cfg/{}_{}_play.yaml'.format(env, task)
    args['steps'] = 0
    args['trials'] = 1
    args['test_interval'] = 0
    args['normalize_observations'] = False
    args['normalize_returns'] = False
    args['batch_norm'] = True

    args['output'] = misc + '{}_{}_play-mp{}'.format(env, task, mp)
    t = task[0].lower()
    if t == 'b':
        stage = stage_names[1]
    elif t == 'w':
        stage = stage_names[2]
    else:
        raise ValueError('incorrect task ' + task)
    args['load_file'] = policies + 'ddpg-exp1_two_stage_leo_ga_{task}-g0001-mp{mp}-{stage}'.format(task=t, mp=mp, stage=stage)
    args['compare_with'] = policies + 'ddpg-exp1_two_stage_leo_ga_b-g0001-mp{mp}-01_balancing-last'.format(mp=mp)
    args['trajectory'] = trajectories + '{}_{}-mp{}'.format(env, task, mp)

    args['env_timestep'] = 0.03

    # Run actual script.
    args['save'] = False
    cfg_run(**args)

    if task == 'walking':
        os.rename('aux_leo.csv', 'aux_leo-mp{}.csv'.format(mp))


if __name__ == '__main__':
    main()