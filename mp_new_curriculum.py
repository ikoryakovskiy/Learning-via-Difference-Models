#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from multiprocessing import cpu_count
from ddpg import parse_args
from cl_learning import Helper, do_multiprocessing_pool
from cl_main import cl_run
import random

def main():
    args = parse_args()

    if args['cores']:
        cores = min(cpu_count(), args['cores'])
    else:
        cores = min(cpu_count(), 16)
    print('Using {} cores.'.format(cores))

    runs = range(16)

    best  = '-0.30174284 -0.66515505 -0.09394964  0.43391448  1.25390471  0.54831222  0.27742692 -0.38982498'
    avg5  = '-0.3837265  -0.05341943 -0.05598724  0.44945501  0.30258128  0.53435247  0.41192509 -0.36213952'
    avg10 = '0.08924442 -0.17804338 -0.17867106  0.4035898   0.08705742  0.26096178  0.28231048 -0.21078831'

    mp_cfgs = []
    mp_cfgs += do_network_based(args, cores, name='ddpg-best', nn_params=best, runs=runs)
    mp_cfgs += do_network_based(args, cores, name='ddpg-avg5', nn_params=avg5, runs=runs)
    mp_cfgs += do_network_based(args, cores, name='ddpg-avg10', nn_params=avg10, runs=runs)

    mp_cfgs += do_steps_based(args, cores, name='ddpg-bbw', steps=(20000, 30000, 250000), runs=runs)
    mp_cfgs += do_steps_based(args, cores, name='ddpg-bw',  steps=(   -1, 50000, 250000), runs=runs)
    mp_cfgs += do_steps_based(args, cores, name='ddpg-w',   steps=(   -1,    -1, 300000), runs=runs)

    # Run all scripts at once
    random.shuffle(mp_cfgs)
    do_multiprocessing_pool(cores, mp_cfgs)
    #config, tasks, starting_task = mp_cfgs[0]
    #cl_run(tasks, starting_task, **config)

def do_steps_based(args, cores, name, steps, runs):
    args['cl_on'] = 0
    args['steps'] = steps

    # Parameters
    starting_task = 'balancing_tf'
    tasks = {
            'balancing_tf': 'cfg/leo_balancing_tf.yaml',
            'balancing':    'cfg/leo_balancing.yaml',
            'walking':      'cfg/leo_walking.yaml'
            }
    hp = Helper(args, 'cl', name, tasks, starting_task, cores, use_mp=True)

    # Weights of the NN
    solutions = [None]*len(runs)
    begin = runs[0]

    mp_cfgs = hp.gen_cfg(solutions, 1, begin=begin)
    return mp_cfgs


def do_network_based(args, cores, name, nn_params, runs):
    args['cl_on'] = 3
    args['cl_structure'] = '_2'
    args['perf_td_error'] = True
    args['perf_l2_reg'] = True
    args['steps'] = 300000
    args['cl_depth'] = 1

    # Parameters
    starting_task = 'balancing_tf'
    tasks = {
            'balancing_tf': 'cfg/leo_balancing_tf.yaml',
            'balancing':    'cfg/leo_balancing.yaml',
            'walking':      'cfg/leo_walking.yaml'
            }
    hp = Helper(args, 'cl', name, tasks, starting_task, cores, use_mp=True)

    # Weights of the NN
    solution = [float(i) for i in nn_params.split()]
    solutions = [solution]*len(runs)
    begin = runs[0]

    mp_cfgs = hp.gen_cfg(solutions, 1, begin=begin)
    return mp_cfgs


######################################################################################
if __name__ == "__main__":
    main()
