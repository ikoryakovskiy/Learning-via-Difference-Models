#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddpg import parse_args
from cl_main import cl_run
from cl_learning import Helper
args = parse_args()

args['rb_min_size'] = 1000
args['reach_return'] = 526.0
args['default_damage'] = 4132.00
args['perf_td_error'] = True
args['perf_l2_reg'] = True
args['steps'] = 300000
args["rb_max_size"] = args['steps']
#args["cl_batch_norm"] = True
#args['cl_structure'] = 'ffcritic:fc_relu_4;fc_relu_3;fc_relu_3'
args["cl_batch_norm"] = False
args['cl_structure'] = 'rnnc:gru_tanh_6_dropout;fc_linear_3'
args["cl_stages"] = "balancing_tf;balancing;walking:monotonic"
args['cl_depth'] = 2
args['cl_pt_shape'] = (args['cl_depth'],3)
args['test_interval'] = 30


#args["cl_target"] = True
export_names = "eq_curriculum_network_depth_" + str(args['cl_depth'])
nn_params = (export_names, "{}_stat.pkl".format(export_names))
args["cl_pt_load"] = nn_params[1]


# Parameters
tasks = {
        'balancing_tf': 'cfg/leo_balancing_tf.yaml',
        'balancing':    'cfg/leo_balancing.yaml',
        'walking':      'cfg/leo_walking.yaml'
        }
starting_task = 'balancing_tf'
hp = Helper(args, 'cl', 'ddpg', tasks, starting_task, 1, use_mp=False)

# Run actual script.
config, tasks, starting_task = hp.gen_cfg([None], 1)[0]
config["cl_load"] = nn_params[0]
cl_run(tasks, starting_task, **config)
