#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from multiprocessing import cpu_count
from ddpg import parse_args
from cl_learning import Helper, prepare_multiprocessing, do_multiprocessing_pool
import random
import yaml, io
from cl_main import cl_run

def main():
    args = parse_args()

    if args['cores']:
        cores = min(cpu_count(), args['cores'])
    else:
        cores = min(cpu_count(), 16)
    print('Using {} cores.'.format(cores))

    # Parameters
    runs = range(3)
    exp_name = "ddpg-exp1_two_stage"

    starting_task = 'balancing_tf'
    misc = {'starting_task':starting_task, 'runs':runs}
    mp_cfgs = []

    keep_samples = False

    # Leo
    tasks = {
            'balancing_tf': 'cfg/leo_balancing_tf.yaml',
            'balancing':    'cfg/leo_balancing.yaml',
            'walking':      'cfg/leo_walking.yaml'
            }
    bsteps = 50000
    steps  = 300000
    reassess_for = 'walking_300_-1.5'
    args['rb_max_size'] = steps if keep_samples else steps - bsteps
    args['env_timestep'] = 0.03
    cl_options = {'balancing_tf': '', 'balancing': '', 'walking': 'nnload_rbload'}
    mp_cfgs += create_tasks(args, cores, exp_name+'_leo', bsteps, steps, reassess_for, tasks, cl_options, **misc)

    # Hopper
    tasks = {
        'balancing_tf': 'RoboschoolHopperBalancingGRL-v1',
        'balancing':    'RoboschoolHopperBalancingGRL-v1',
        'walking':      'RoboschoolHopperGRL-v1'
        }
    bsteps = 100000
    steps  = 600000
    reassess_for = 'walking_3_-1.5'
    args['rb_max_size'] = steps if keep_samples else steps - bsteps
    args['env_timestep'] = 0.0165
    cl_options = {'balancing_tf': '', 'balancing': '', 'walking': 'nnload_rbload'}
    mp_cfgs += create_tasks(args, cores, exp_name+'_hopper', bsteps, steps, reassess_for, tasks, cl_options, **misc)

    # HalfCheetah
    tasks = {
        'balancing_tf': 'RoboschoolHalfCheetahBalancingGRL-v1',
        'balancing':    'RoboschoolHalfCheetahBalancingGRL-v1',
        'walking':      'RoboschoolHalfCheetahGRL-v1'
        }
    bsteps = 100000
    steps  = 600000
    reassess_for = 'walking_3_-1.5'
    args['rb_max_size'] = steps if keep_samples else steps - bsteps
    args['env_timestep'] = 0.0165
    cl_options = {'balancing_tf': '', 'balancing': '', 'walking': 'nnload'}
    mp_cfgs += create_tasks(args, cores, exp_name+'_halfcheetah', bsteps, steps, reassess_for, tasks, cl_options, **misc)

    # Walker2d
    tasks = {
        'balancing_tf': 'RoboschoolWalker2dBalancingGRL-v1',
        'balancing':    'RoboschoolWalker2dBalancingGRL-v1',
        'walking':      'RoboschoolWalker2dGRL-v1'
        }
    bsteps = 200000
    steps  = 700000
    reassess_for = 'walking_3_-1.5'
    args['rb_max_size'] = steps if keep_samples else steps - bsteps
    args['env_timestep'] = 0.0165
    cl_options = {'balancing_tf': '', 'balancing': '', 'walking': 'nnload'}
    mp_cfgs += create_tasks(args, cores, exp_name+'_walker2d', bsteps, steps, reassess_for, tasks, cl_options, **misc)

    # DBG: export configuration
    export_cfg(mp_cfgs)

    # Run all scripts at once
    random.shuffle(mp_cfgs)
    prepare_multiprocessing()
    do_multiprocessing_pool(cores, mp_cfgs)
    #config, tasks, starting_task = mp_cfgs[0]
    #cl_run(tasks, starting_task, **config)


def create_tasks(args, cores, exp_name, bsteps, steps, reassess_for, tasks, cl_options={}, **misc):
    mp_cfgs = []

    options = {'balancing_tf': '', 'balancing': '', 'walking': ''}
    mp_cfgs += do_steps_based(args, cores, name=exp_name+'_ga_w',   steps=(-1,  -1, steps), options=options, tasks=tasks, **misc)

    options = {'balancing_tf': '', 'balancing': '', 'walking': ''}
    mp_cfgs += do_steps_based(args, cores, name=exp_name+'_ga_b',   steps=(-1,  bsteps, -1), options=options, tasks=tasks, **misc)

    # curriculum option
    wsteps = steps - bsteps
    mp_cfgs += do_steps_based(args, cores, name=exp_name+'_ga_bw', steps=(-1, bsteps, wsteps), options=cl_options, **misc)

    return mp_cfgs


def do_steps_based(base_args, cores, name, steps, runs, options=None, tasks={}, starting_task=''):
    args = base_args.copy()
    args['steps'] = steps

    if options:
        suffix = ''
        if options['balancing_tf']:
            suffix += '1_' + options['balancing_tf']
        if options['balancing']:
            suffix += '2_' + options['balancing']
        if options['walking']:
            suffix += '3_' + options['walking']
        if suffix:
            name += '-' + suffix
        args['options'] = options

    hp = Helper(args, 'cl', name, tasks, starting_task, cores, use_mp=True)

    # generate configurations
    solutions = [None]*len(runs)
    begin = runs[0]

    mp_cfgs = hp.gen_cfg(solutions, 1, begin=begin)

    # add file to save trajectories
    for cfg in mp_cfgs:
        cfg[0]['trajectory'] = cfg[0]['output']
    return mp_cfgs


def export_cfg(mp_cfgs):
    for cfg in mp_cfgs:
        config, tasks, starting_task = cfg
        with io.open(config['output']+'.yaml', 'w', encoding='utf8') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

######################################################################################
if __name__ == "__main__":
    main()
