#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import yaml, collections
import argparse
import os
import ddpg_params
from main_ddpg import start

def dict_representer(dumper, data):
    return dumper.represent_dict(data.iteritems())


######################################################################################
def dict_constructor(loader, node):
    return collections.OrderedDict(loader.construct_pairs(node))


######################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('cfg', nargs='?', default="../grl/qt-build/cfg/leo/drl/rbdl_walking_play.yaml")
    parser.add_argument('cfg', nargs='?', default="../grl/qt-build/cfg/leo/drl/rbdl_balancing_play.yaml")
    args = parser.parse_args()

    # create a copy of cfg with DDPG learning parameters in temporary location
    loc = "tmp"
    if not os.path.exists(loc):
        os.makedirs(loc)
    new_cfg = "{}/{}".format(loc, os.path.basename(args.cfg))

    # for walking with yaml files
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
    yaml.add_representer(collections.OrderedDict, dict_representer)
    yaml.add_constructor(_mapping_tag, dict_constructor)

    # append ddpg_param if missing
    with open(args.cfg, 'r') as f:
        conf = yaml.load(f)
        if not "ddpg_param" in conf:
            conf['ddpg_param'] = ddpg_params.init()
            conf['ddpg_param']['learning']["ou_sigma"] = 0
            conf['ddpg_param']['learning']["ou_theta"] = 1
            conf['ddpg_param']['learning']["actor_learning_rate"] = 0
            conf['ddpg_param']['learning']["critic_learning_rate"] = 0
            with open(new_cfg, 'w') as f:
                yaml.dump(conf, f)

    # Use newly created configuration
    start(new_cfg)
