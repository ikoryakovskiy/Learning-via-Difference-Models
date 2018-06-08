#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddpg import parse_args
from cl_main import cl_run
from cl_learning import Helper
args = parse_args()

args['rb_min_size'] = 1000
args['reach_return'] = 1422.66
args['default_damage'] = 4035.00
args['perf_td_error'] = True
args['perf_l2_reg'] = True
args['steps'] = 300000
args["cl_batch_norm"] = False
args['cl_structure'] = 'rnnc:gru_tanh_6_dropout;fc_linear_3'
args['cl_stages'] = 'balancing_tf;balancing;walking:monotonic'
args['cl_depth'] = 2
args['cl_pt_shape'] = (2,3)

nn_params=("short_curriculum_network", "short_curriculum_network_stat.pkl")

#args["cl_target"] = True
args["cl_pt_load"] = nn_params[1]

# Parameters
perturbed = True
if not perturbed:
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
hp = Helper(args, 'cl', 'ddpg', tasks, starting_task, 1, use_mp=False)

# Run actual script.
config, tasks, starting_task = hp.gen_cfg([None], 1)[0]
config["cl_load"] = nn_params[0]
cl_run(tasks, starting_task, **config)
