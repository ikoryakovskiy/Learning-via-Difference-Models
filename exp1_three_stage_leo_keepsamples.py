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
    exp_name = "ddpg-exp1_three_stage"

    tasks = {
            'balancing_tf': 'cfg/leo_balancing_tf.yaml',
            'balancing':    'cfg/leo_balancing.yaml',
            'walking':      'cfg/leo_walking.yaml'
            }


    starting_task = 'balancing_tf'
    misc = {'tasks':tasks, 'starting_task':starting_task, 'runs':runs}

    mp_cfgs = []

    steps = 300000
    args['rb_max_size'] = steps
    args['cl_keep_samples'] = True
    options = {'balancing_tf': '', 'balancing': '', 'walking': ''}
    mp_cfgs += do_steps_based(args, cores, name=exp_name,   steps=(-1,  -1, steps), options=options, **misc)

    btfsteps = 20000
    bsteps = 30000
    wsteps = steps - bsteps - btfsteps
    args['rb_max_size'] = steps

    #for bb in ['nnload', 'rbload', 'nnload_rbload']:
    for bb in ['nnload']:

        options = {'balancing_tf': '', 'balancing': bb, 'walking': 'nnload'}
        mp_cfgs += do_steps_based(args, cores, name=exp_name, steps=(btfsteps, bsteps, wsteps), options=options, **misc)

        options = {'balancing_tf': '', 'balancing': bb, 'walking': 'nnload_rbload'}
        mp_cfgs += do_steps_based(args, cores, name=exp_name, steps=(btfsteps, bsteps, wsteps), options=options, **misc)

        options = {'balancing_tf': '', 'balancing': bb, 'walking': 'rbload'}
        mp_cfgs += do_steps_based(args, cores, name=exp_name, steps=(btfsteps, bsteps, wsteps), options=options, **misc)

    # DBG: export configuration
    export_cfg(mp_cfgs)

    # Run all scripts at once
    random.shuffle(mp_cfgs)
    prepare_multiprocessing()
    do_multiprocessing_pool(cores, mp_cfgs)
    #config, tasks, starting_task = mp_cfgs[0]
    #cl_run(tasks, starting_task, **config)


def do_steps_based(base_args, cores, name, steps, runs, options=None, tasks={}, starting_task=''):
    args = base_args.copy()
    args['steps'] = steps

    if options:
        suffix = ''
        if options['balancing_tf']:
            suffix += '1_' + options['balancing_tf'] + '_'
        if options['balancing']:
            suffix += '2_' + options['balancing'] + '_'
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
