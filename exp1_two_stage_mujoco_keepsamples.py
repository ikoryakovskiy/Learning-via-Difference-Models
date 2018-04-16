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
    runs = range(16)
    exp_name = "ddpg-exp1_two_stage"

    starting_task = 'balancing_tf'
    misc = {'starting_task':starting_task, 'runs':runs}
    mp_cfgs = []

    keep_samples = True

    # Hopper
    tasks = {
        'balancing_tf': 'RoboschoolHopperBalancingGRL-v1',
        'balancing':    'RoboschoolHopperBalancingGRL-v1',
        'walking':      'RoboschoolHopperGRL-v1'
        }
    bsteps = 100000
    steps  = 600000
    reassess_for = 'walking_181.8182_-1.5'
    args['rb_max_size'] = steps if keep_samples else steps - bsteps
    mp_cfgs += create_tasks(args, cores, exp_name+'_hopper', bsteps, steps, reassess_for, tasks, **misc)

    # HalfCheetah
    tasks = {
        'balancing_tf': 'RoboschoolHalfCheetahBalancingGRL-v1',
        'balancing':    'RoboschoolHalfCheetahBalancingGRL-v1',
        'walking':      'RoboschoolHalfCheetahGRL-v1'
        }
    bsteps = 100000
    steps  = 600000
    reassess_for = 'walking_181.8182_-1.5'
    args['rb_max_size'] = steps if keep_samples else steps - bsteps
    mp_cfgs += create_tasks(args, cores, exp_name+'_halfcheetah', bsteps, steps, reassess_for, tasks, **misc)

    # Walker2d
    tasks = {
        'balancing_tf': 'RoboschoolWalker2dBalancingGRL-v1',
        'balancing':    'RoboschoolWalker2dBalancingGRL-v1',
        'walking':      'RoboschoolWalker2dGRL-v1'
        }

    bsteps = 200000
    steps  = 700000
    reassess_for = 'walking_121.2121_-1.5'
    args['rb_max_size'] = steps if keep_samples else steps - bsteps
    mp_cfgs += create_tasks(args, cores, exp_name+'_walker2d', bsteps, steps, reassess_for, tasks, **misc)

    # DBG: export configuration
    export_cfg(mp_cfgs)

    # Run all scripts at once
    random.shuffle(mp_cfgs)
    prepare_multiprocessing()
    do_multiprocessing_pool(cores, mp_cfgs)
    #config, tasks, starting_task = mp_cfgs[0]
    #cl_run(tasks, starting_task, **config)


def create_tasks(args, cores, exp_name, bsteps, steps, reassess_for, tasks, **misc):
    wsteps = steps - bsteps

    mp_cfgs = []

    options = {'balancing_tf': '', 'balancing': '', 'walking': ''}
    mp_cfgs += do_steps_based(args, cores, name=exp_name,   steps=(-1,  -1, steps), options=options, tasks=tasks, **misc)

    options = {'balancing_tf': '', 'balancing': '', 'walking': 'nnload'}
    mp_cfgs += do_steps_based(args, cores, name=exp_name, steps=(-1, bsteps, wsteps), options=options, tasks=tasks, **misc)

#    options = {'balancing_tf': '', 'balancing': '', 'walking': 'nnload_rbload'}
#    mp_cfgs += do_steps_based(args, cores, name=exp_name, steps=(-1, bsteps, wsteps), options=options, tasks=tasks, **misc)
#
#    options = {'balancing_tf': '', 'balancing': '', 'walking': 'nnload_rbload_re_{}'.format(reassess_for)}
#    mp_cfgs += do_steps_based(args, cores, name=exp_name, steps=(-1, bsteps, wsteps), options=options, tasks=tasks, **misc)
#
#    options = {'balancing_tf': '', 'balancing': '', 'walking': 'rbload'}
#    mp_cfgs += do_steps_based(args, cores, name=exp_name, steps=(-1, bsteps, wsteps), options=options, tasks=tasks, **misc)
#
#    options = {'balancing_tf': '', 'balancing': '', 'walking': 'rbload_re_{}'.format(reassess_for)}
#    mp_cfgs += do_steps_based(args, cores, name=exp_name, steps=(-1, bsteps, wsteps), options=options, tasks=tasks, **misc)

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
    return mp_cfgs


def export_cfg(mp_cfgs):
    for cfg in mp_cfgs:
        config, tasks, starting_task = cfg
        with io.open(config['output']+'.yaml', 'w', encoding='utf8') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

######################################################################################
if __name__ == "__main__":
    main()
