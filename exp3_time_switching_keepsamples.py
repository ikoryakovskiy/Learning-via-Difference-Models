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

    tasks = {
            'balancing_tf': 'cfg/leo_balancing_tf.yaml',
            'balancing':    'cfg/leo_balancing.yaml',
            'walking':      'cfg/leo_walking.yaml'
            }


    starting_task = 'balancing_tf'
    misc = {'tasks':tasks, 'starting_task':starting_task, 'runs':runs}

    args['cl_keep_samples'] = True
    options = {'balancing_tf': '', 'balancing': 'nnload', 'walking': 'nnload_rbload'}

    mp_cfgs = []

    # naive switching after achieving the balancing for n number of seconds happening once. 0 means not used
    args['reach_timeout_num'] = 1
    mp_cfgs += do_reach_timeout_based(args, cores, name='ddpg-exp3-three_stage-rb5', reach_timeout=(5.0, 5.0, 0.0), options=options, **misc)
    mp_cfgs += do_reach_timeout_based(args, cores, name='ddpg-exp3-two_stage-rb5', reach_timeout=(-1.0, 5.0, 0.0), options=options, **misc)

    mp_cfgs += do_reach_timeout_based(args, cores, name='ddpg-exp3-three_stage-rb20', reach_timeout=(20.0, 20.0, 0.0), options=options, **misc)
    mp_cfgs += do_reach_timeout_based(args, cores, name='ddpg-exp3-two_stage-rb20', reach_timeout=(-1.0, 20.0, 0.0), options=options, **misc)


    # naive switching after achieving the balancing for n number of seconds happening twice. 0 means not used
    args['reach_timeout_num'] = 2
    mp_cfgs += do_reach_timeout_based(args, cores, name='ddpg-exp3-three_stage-rb55', reach_timeout=(5.0, 5.0, 0.0), options=options, **misc)
    mp_cfgs += do_reach_timeout_based(args, cores, name='ddpg-exp3-two_stage-rb55', reach_timeout=(-1.0, 5.0, 0.0), options=options, **misc)

    mp_cfgs += do_reach_timeout_based(args, cores, name='ddpg-exp3-three_stage-rb2020', reach_timeout=(20.0, 20.0, 0.0), options=options, **misc)
    mp_cfgs += do_reach_timeout_based(args, cores, name='ddpg-exp3-two_stage-rb2020', reach_timeout=(-1.0, 20.0, 0.0), options=options, **misc)

    # DBG: export configuration
    export_cfg(mp_cfgs)

    # Run all scripts at once
    random.shuffle(mp_cfgs)
    prepare_multiprocessing()
    do_multiprocessing_pool(cores, mp_cfgs)
    #config, tasks, starting_task = mp_cfgs[0]
    #cl_run(tasks, starting_task, **config)


def do_reach_timeout_based(base_args, cores, name, reach_timeout, runs, options=None, tasks={}, starting_task=''):
    args = base_args.copy()
    args['reach_timeout'] = reach_timeout
    steps = 300000
    args['steps'] = steps
    args['rb_max_size'] = steps

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

    # Weights of the NN
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
