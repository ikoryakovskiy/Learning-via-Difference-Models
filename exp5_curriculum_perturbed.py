#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from multiprocessing import cpu_count
from ddpg import parse_args
from cl_learning import Helper, prepare_multiprocessing, do_multiprocessing_pool
import random
import yaml, collections, io
from cl_create_models_tasks import PerturbedModelsTasks

def main():
    args = parse_args()

    if args['cores']:
        cores = min(cpu_count(), args['cores'])
    else:
        cores = min(cpu_count(), 16)
    print('Using {} cores.'.format(cores))

    # for working with yaml files
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
    yaml.add_representer(collections.OrderedDict, dict_representer)
    yaml.add_constructor(_mapping_tag, dict_constructor)

    mp_cfgs = []
    starting_task = 'balancing_tf'
    options = {'balancing_tf': '', 'balancing': 'nnload_rbload', 'walking': 'nnload_rbload'}
    #options = {'balancing_tf': '', 'balancing': 'nnload', 'walking': 'nnload_rbload'}


    runs = range(8)

    # create perturbed models of leo
    pmt = PerturbedModelsTasks()
    tasks, names = pmt.generate()

    for task, name in zip(tasks, names):
        misc = {'tasks':task, 'starting_task':starting_task, 'runs':runs}

        args['cl_keep_samples'] = True
        nn_params=("short_curriculum_network", "short_curriculum_network_stat.pkl")
        mp_cfgs += do_network_based_leo(args, cores, name='ddpg-cl_short_'+name, nn_params=nn_params, options=options, **misc)

#        # reach balaning 2 times in 3-task curriculum
#        args['cl_keep_samples'] = True
#        args['reach_timeout_num'] = 5
#        mp_cfgs += do_reach_timeout_based(args, cores, name='ddpg-rb55555-tuned-'+name, reach_timeout=(-1.0, 5.0, 0.0), options=options, **misc)
#        args['reach_timeout_num'] = 0
#        args['cl_keep_samples'] = False

        # direct learning
        #mp_cfgs += do_steps_based(args, cores, name='ddpg-direct-'+name, steps=(-1,  -1, 300000), options=options, **misc)

        # regular with keepsamples = True
        args['cl_keep_samples'] = True
        mp_cfgs += do_steps_based(args, cores, name='ddpg-steps_based-ks1-tuned-'+name, steps=(1833, 45000, 253167), options=options, **misc)
    # \prepare



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


def do_network_based_mujoco(base_args, cores, name, nn_params, runs, tasks, starting_task):
    args = base_args.copy()
    args['env_td_error_scale'] = 600.0
    args['env_timeout'] = 16.5

    args['rb_min_size'] = 1000
    args['default_damage'] = 4035.00
    args['perf_td_error'] = True
    args['perf_l2_reg'] = True
    args['steps'] = 700000
    args["cl_batch_norm"] = False
    args['cl_structure'] = 'rnnc:gru_tanh_6_dropout;fc_linear_3'
    args['cl_stages'] = 'balancing_tf;balancing;walking:monotonic'
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


def do_network_based_leo(base_args, cores, name, nn_params, runs, options=None, tasks={}, starting_task=''):
    args = base_args.copy()
    args['rb_min_size'] = 1000
    args['default_damage'] = 4035.00
    args['perf_td_error'] = True
    args['perf_l2_reg'] = True
    args['steps'] = 300000
    args["cl_batch_norm"] = False
    args['cl_structure'] = 'rnnc:gru_tanh_6_dropout;fc_linear_3'
    args['cl_stages'] = 'balancing_tf;balancing;walking:monotonic'
    args['cl_depth'] = 2
    args['cl_pt_shape'] = (2,3)
    args["cl_pt_load"] = nn_params[1]
    cl_load = nn_params[0]

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

def remove_viz(conf):
    """Remove everything in conf related to visualization"""
    if "visualize" in conf['experiment']['environment']:
        conf['experiment']['environment']['visualize'] = 0
    if "target_env" in conf['experiment']['environment']:
    	if "visualize" in conf['experiment']['environment']['target_env']:
        	conf['experiment']['environment']['target_env']['visualize'] = 0
    if "visualizer" in conf:
            del conf["visualizer"]
    if "visualization" in conf:
            del conf["visualization"]
    if "visualization2" in conf:
            del conf["visualization2"]
    return conf
######################################################################################

def dict_representer(dumper, data):
  return dumper.represent_dict(data.items())
######################################################################################

def dict_constructor(loader, node):
  return collections.OrderedDict(loader.construct_pairs(node))

######################################################################################
######################################################################################

if __name__ == "__main__":
    main()
