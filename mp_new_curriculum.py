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
    runs = range(16, 32)
    #model = 'perturbed'
    model = 'idealized'

    if model == 'idealized':
        tasks = {
                'balancing_tf': 'cfg/leo_balancing_tf.yaml',
                'balancing':    'cfg/leo_balancing.yaml',
                'walking':      'cfg/leo_walking.yaml'
                }
    else:
        tasks = {
                'balancing_tf': 'cfg/leo_perturbed_balancing_tf.yaml',
                'balancing':    'cfg/leo_perturbed_balancing.yaml',
                'walking':      'cfg/leo_perturbed_walking.yaml'
                }

    starting_task = 'balancing_tf'
    misc = {'tasks':tasks, 'starting_task':starting_task, 'runs':runs}

    mp_cfgs = []
    nn_params=("long_curriculum_network", "long_curriculum_network_stat.pkl")
    mp_cfgs += do_network_based(args, cores, name='ddpg-cl_long', nn_params=nn_params, **misc)

    nn_params=("short_curriculum_network", "short_curriculum_network_stat.pkl")
    mp_cfgs += do_network_based(args, cores, name='ddpg-cl_short', nn_params=nn_params, **misc)

    mp_cfgs += do_steps_based(args, cores, name='ddpg-bbw', steps=(20000, 30000, 250000), **misc)
    mp_cfgs += do_steps_based(args, cores, name='ddpg-bw',  steps=(   -1, 50000, 250000), **misc)
    mp_cfgs += do_steps_based(args, cores, name='ddpg-w',   steps=(   -1,    -1, 300000), **misc)

    # naive switching after achieving the balancing for n number of seconds happening twice. 0 means not used
    mp_cfgs += do_reach_timeout_based(args, cores, name='ddpg-rb55', reach_timeout=(5.0, 5.0, 0.0), **misc)
    mp_cfgs += do_reach_timeout_based(args, cores, name='ddpg-rb5', reach_timeout=(-1.0, 5.0, 0.0), **misc)

    mp_cfgs += do_reach_timeout_based(args, cores, name='ddpg-rb2020', reach_timeout=(20.0, 20.0, 0.0), **misc)
    mp_cfgs += do_reach_timeout_based(args, cores, name='ddpg-rb20', reach_timeout=(-1.0, 20.0, 0.0), **misc)

    # DBG: export configuration
    export_cfg(mp_cfgs)


    # Run all scripts at once
    random.shuffle(mp_cfgs)
    prepare_multiprocessing()
    do_multiprocessing_pool(cores, mp_cfgs)
    #config, tasks, starting_task = mp_cfgs[0]
    #cl_run(tasks, starting_task, **config)


def do_steps_based(base_args, cores, name, steps, runs, tasks, starting_task):
    args = base_args.copy()
    args['steps'] = steps

    hp = Helper(args, 'cl', name, tasks, starting_task, cores, use_mp=True)

    # Weights of the NN
    solutions = [None]*len(runs)
    begin = runs[0]

    mp_cfgs = hp.gen_cfg(solutions, 1, begin=begin)
    return mp_cfgs


def do_reach_timeout_based(base_args, cores, name, reach_timeout, runs, tasks, starting_task):
    args = base_args.copy()
    args['reach_timeout'] = reach_timeout
    args['steps'] = 300000

    hp = Helper(args, 'cl', name, tasks, starting_task, cores, use_mp=True)

    # Weights of the NN
    solutions = [None]*len(runs)
    begin = runs[0]

    mp_cfgs = hp.gen_cfg(solutions, 1, begin=begin)
    return mp_cfgs


def do_network_based(base_args, cores, name, nn_params, runs, tasks, starting_task):
    args = base_args.copy()
    args['rb_min_size'] = 1000
    args['default_damage'] = 4035.00
    args['perf_td_error'] = True
    args['perf_l2_reg'] = True
    args['steps'] = 300000
    args["cl_batch_norm"] = False
    args['cl_structure'] = 'rnnc:gru_tanh_6_dropout;fc_linear_3'
    args['cl_depth'] = 2
    args['cl_pt_shape'] = (2,3)
    args["cl_pt_load"] = nn_params[1]
    cl_load = nn_params[0]

    hp = Helper(args, 'cl', name, tasks, starting_task, cores, use_mp=True)

    # Weights of the NN
    solutions = [None]*len(runs)
    begin = runs[0]

    mp_cfgs = hp.gen_cfg(solutions, 1, begin=begin)
    mp_cfgs_new = []
    for cfg in mp_cfgs:
        config, tasks, starting_task = cfg
        copy_config = config.copy()
        copy_config["cl_load"] = cl_load
        mp_cfgs_new.append( (copy_config, tasks, starting_task) )
    return mp_cfgs_new



def export_cfg(mp_cfgs):
    for cfg in mp_cfgs:
        config, tasks, starting_task = cfg
        with io.open(config['output']+'.yaml', 'w', encoding='utf8') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

######################################################################################
if __name__ == "__main__":
    main()
