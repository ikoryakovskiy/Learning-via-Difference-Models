#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import time
from main_ddpg import start
import yaml
import tensorflow as tf
from train_model import training
from train_model import prediction
import pickle
import numpy as np
import os.path
import sys
from collections import OrderedDict
import global_params
import multiprocessing

from difference_model import DifferenceModel


def foo():
    path = os.path.dirname(os.path.abspath(__file__))

    # Run the training on the ideal model phase
    # cfg = "{}/leo_rbdl_zmq_drl.yaml".format(path)
    # a = start(cfg)
    # time.sleep(5)
    # #

    # No of policy iterations
    for ii in range(1, 10):
    #
        # No of runs of one policy
        for i in range(1):
            #
            # Run the trained policy on a real model
            global_params.ou_sigma = 0.1
            global_params.ou_theta = 0.15
            d = {'transitions': {'load': 0, 'save': 1, 'save_filename': 'saved_data-perturbed-{}'.format(ii), 'buffer_size': 5000},
                 'difference_model': 0}
            with open('config.yaml', 'w') as yaml_file:
                yaml.dump(d, yaml_file, default_flow_style=False)
            cfg = "{}/leo_rbdl_zmq_drl_2.yaml".format(path)
            new_cfg = rl_run_rbdl_agent(cfg, ii - 1)

            start(new_cfg)
            time.sleep(2)

            # Run the transitions on the original model
            d = {'transitions': {'load': 1, 'load_filename': 'saved_data-perturbed-{}'.format(ii), 'save': 1,
                                   'save_filename': 'saved_data-original-{}'.format(ii), 'buffer_size': 5000},
                 'difference_model': 0}
            with open('config.yaml', 'w') as yaml_file:
                yaml.dump(d, yaml_file, default_flow_style=False)
            cfg = "{}/leo_rbdl_zmq_drl_3.yaml".format(path)
            start(cfg)

            # Train a new difference model or update one
            with tf.Graph().as_default() as diff_model:
                model = DifferenceModel(24, 18)
                with tf.Session(graph=diff_model) as sess:
                    if i == 0 and ii == 1:
                        d = {'difference_model': 0}
                    else:
                        d = {'difference_model': 1}
                    with open('config.yaml', 'w') as yaml_file:
                        yaml.dump(d, yaml_file, default_flow_style=False)

                    perturbed_files = ['saved_data-perturbed-{}'.format(b) for b in range(1, ii+1)]
                    ideal_files = ['saved_data-original-{}'.format(b) for b in range(1, ii+1)]
                    print perturbed_files
                    model = training(sess, model, perturbed_files, ideal_files, 24, 18, 300)


        # Training the policy with the difference model included
        global_params.ou_sigma = 0.12
        global_params.ou_theta = 0.15
        global_params.actor_learning_rate = 0.0001
        global_params.critic_learning_rate = 0.001
        iterations = 0
        while not global_params.learning_success and iterations != 1:
            if ii == 1:
                d = {'replay_buffer': {'load': 0, 'save': 1, 'save_filename': 'saved_replay_buffer',
                                       'buffer_size': 100000}, 'difference_model': 0}
                with open('config.yaml', 'w') as yaml_file:
                    yaml.dump(d, yaml_file, default_flow_style=False)
            else:
                d = {'replay_buffer': {'load': 1, 'load_filename': 'saved_replay_buffer', 'save': 0, 'save_filename': 'saved_replay_buffer',
                                       'buffer_size': 100000}, 'difference_model': 0}
                with open('config.yaml', 'w') as yaml_file:
                    yaml.dump(d, yaml_file, default_flow_style=False)
            cfg = "{}/leo_rbdl_zmq_drl.yaml".format(path)
            new_cfg = rl_run_rbdl_agent(cfg, ii - 1)
            start(new_cfg, ii)

            time.sleep(5)

            # Running the learned policy on the difference model on the perturbed system to see if it works
            global_params.test_run_on_model = 1

            d = {'replay_buffer': {'load': 0, 'save': 0, 'buffer_size': 2000}, 'difference_model': 0}
            with open('config.yaml', 'w') as yaml_file:
                yaml.dump(d, yaml_file, default_flow_style=False)
            cfg = "{}/leo_rbdl_zmq_drl_real.yaml".format(path)
            new_cfg = rl_run_rbdl_agent(cfg, ii)
            start(new_cfg)
            global_params.test_run_on_model = 0

            iterations += 1

        global_params.learning_success = 0

        if global_params.learning_success:
            print ("Entire training was successful")
            break


def rl_run_rbdl_agent(config, iteration=0, c=0):
    fname = os.path.splitext(config)[0]
    new_config = "{}-{}.yaml".format(fname, iteration)
    conf = read_cfg_divyam(config)

    if "output" in conf['experiment']:
        output = conf['experiment']['output']
        conf['experiment']['output'] = "{}-{}".format(output, iteration)
        # Change the binding ports for each config file
    if "load_file" in conf['experiment']:
        if iteration != 0:
            conf["experiment"]["load_file"] = "model-leo-rbdl-with-diff-{}.ckpt".format(iteration)

    write_cfg(new_config, conf)

    return new_config


def write_cfg(outCfg, conf):
    """Write configuration file"""
    # create local yaml configuration file
    outfile = file(outCfg, 'w')
    ordered_dump(conf, outfile, yaml.SafeDumper)
    outfile.close()


def read_cfg_divyam(cfg):
    """Read configuration file"""
    # check if file exists
    yfile = '%s' % cfg
    if os.path.isfile(yfile) == False:
        print 'File %s not found' % yfile
        sys.exit()

    # open configuration
    stream = file(yfile, 'r')
    conf = ordered_load(stream, yaml.SafeLoader)
    stream.close()
    return conf


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def main():
    global_params.init()
    foo()


if __name__ == '__main__':
    main()
