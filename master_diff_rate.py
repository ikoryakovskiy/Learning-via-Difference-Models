#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import time
from main_ddpg_diff_rate import start
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
    for ii in range(1,2):

        # No of runs of one policy
        # Training the policy with the difference model included
        global_params.ou_sigma = 0.2
        global_params.ou_theta = 0.15
        global_params.actor_learning_rate = 0.00012
        global_params.critic_learning_rate = 0.0012
        RUNS = 5
        while global_params.reward == 0:
            d = {'replay_buffer': {'load': 0, 'save': 0, 'buffer_size': 20000}, 'difference_model': 0}
            with open('config.yaml', 'w') as yaml_file:
                yaml.dump(d, yaml_file, default_flow_style=False)
            cfg = ["/leo_drl_server/leo_rbdl_zmq_drl_diff_rate.yaml"]
            new_cfg = rl_run_zmqagent(cfg, range(RUNS))

            pool = multiprocessing.Pool(RUNS)
            pool.map(start, new_cfg)
            pool.close()

            # start(new_cfg, ii)

        global_params.reward = 0
        time.sleep(5)
        global_params.test_run_on_model = 1
	global_params.learning_success = 0
        # Running the learned policy on the difference model on the perturbed system to see if it works
        

def rl_run_zmqagent(list_of_cfgs, runs):
    list_of_new_cfgs = []

    for cfg in list_of_cfgs:
        conf = read_cfg_divyam(cfg)

        fname, fext = os.path.splitext(cfg.replace("/", "_"))

        for run in runs:
            # create local filename
            list_of_new_cfgs.append("{}-mp{}{}".format(fname, run, fext))

            # modify options
        #    conf['experiment']['environment']['xml'] = "../grl/addons/cfg/leo/xm430_210_vc_leo_walk_ankle_bound.xml"
            conf['experiment']['output'] = "{}-mp-rbdl{}".format(fname, run)
            if "exporter" in conf['experiment']['environment']:
                conf['experiment']['environment']['exporter']['file'] = "{}-mp-rbdl{}".format(fname, run)
                # Change the binding ports for each config file
            if "communicator" in conf['experiment']['agent']:
                conf['experiment']['agent']['communicator']['addr'] = "tcp://localhost:555{}".format(run)
                conf['experiment']['test_agent']['communicator']['addr'] = "tcp://localhost:555{}".format(run)
            if "learning_rate" in conf['experiment']:
                conf['experiment']['learning_rate']= 0.001 + 0.0002*run
            if "difference_model" in conf['experiment']:
                conf['experiment']['difference_model']= 1
            write_cfg(list_of_new_cfgs[-1], conf)

        return list_of_new_cfgs

def rl_run_rbdl_agent(config, iteration = 0):
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
