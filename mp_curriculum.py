from __future__ import division
import multiprocessing
import os
import os.path
import sys
import yaml, collections
from time import sleep
import argparse
import itertools
import signal
import random
import socket
from contextlib import closing
from datetime import datetime
from main_ddpg import start
import ddpg_params

counter_lock = multiprocessing.Lock()
cores = 0
random.seed(datetime.now())

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument('-c', '--cores', type=int, help='specify maximum number of cores')
    args = parser.parse_args()
    if args.cores:
        args.cores = min(multiprocessing.cpu_count(), args.cores)
    else:
        args.cores = min(multiprocessing.cpu_count(), 32)
    print('Using {} cores.'.format(args.cores))

    # for walking with yaml files
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
    yaml.add_representer(collections.OrderedDict, dict_representer)
    yaml.add_constructor(_mapping_tag, dict_constructor)

    port = 5557

    # Parameters
    runs = range(10)

    options = []
    for r in itertools.product(runs): options.append(r)
    options = [flatten(tupl) for tupl in options]

    configs = [
                "leo/drl/rbdl_balancing.yaml"
                "leo/drl/rbdl_balancing_hack.yaml"
              ]
    L, port = rl_run_zero_shot(args, configs, options, port)

    configs = [
                "leo/drl/rbdl_walking.yaml"
              ]
    L1, port = rl_run_zero_shot(args, configs, options, port)

    configs = [
                "leo/drl/rbdl_walking_after_balancing.yaml",
                "leo/drl/rbdl_curriculum_after_balancing.yaml"
              ]
    L2, port = rl_run_curriculum(args, configs, options, port)

    do_multiprocessing_pool(args, L)
    #do_multiprocessing_pool(args, L1+L2)

######################################################################################
def rl_run_zero_shot(args, list_of_cfgs, options, port):
    list_of_new_cfgs = []

    loc = "tmp"
    if not os.path.exists(loc):
        os.makedirs(loc)

    for cfg in list_of_cfgs:
        conf = read_cfg(cfg)
        conf['ddpg_param'] = ddpg_params.init() # adding DDPG configuration

        # after reading cfg can do anything with the name
        fname, fext = os.path.splitext( cfg.replace("/", "_") )

        for o in options:
            str_o = "-".join(map(lambda x : "{:06d}".format(int(round(100000*x))), o[:-1]))  # last element in 'o' is reserved for mp
            if not str_o:
                str_o += "mp{}".format(o[-1])
            else:
                str_o += "-mp{}".format(o[-1])
            print("Generating parameters: {}".format(str_o))

            # create local filename
            list_of_new_cfgs.append( "{}/{}-{}{}".format(loc, fname, str_o, fext) )

            # select port to use
            port = port_select(port)

            conf['experiment']['output'] = "{}-{}".format(fname, str_o)
            conf['experiment']['agent']['communicator']['addr'] = "tcp://localhost:{}".format(port)
            conf['experiment']['test_agent']['communicator']['addr'] = "tcp://localhost:{}".format(port)

            conf = remove_viz(conf)
            write_cfg(list_of_new_cfgs[-1], conf)

    print(list_of_new_cfgs)

    return list_of_new_cfgs, port


######################################################################################
def rl_run_curriculum(args, list_of_cfgs, options, port):
    list_of_new_cfgs = []

    loc = "tmp"
    if not os.path.exists(loc):
        os.makedirs(loc)

    for cfg in list_of_cfgs:
        conf = read_cfg(cfg)
        conf['ddpg_param'] = ddpg_params.init() # adding DDPG configuration

        # after reading cfg can do anything with the name
        fname, fext = os.path.splitext( cfg.replace("/", "_") )

        for o in options:
            str_o = "-".join(map(lambda x : "{:06d}".format(int(round(100000*x))), o[:-1]))  # last element in 'o' is reserved for mp
            if not str_o:
                str_o += "mp{}".format(o[-1])
            else:
                str_o += "-mp{}".format(o[-1])
            print("Generating parameters: {}".format(str_o))

            # create local filename
            list_of_new_cfgs.append( "{}/{}-{}{}".format(loc, fname, str_o, fext) )

            # select port to use
            port = port_select(port)

            conf['experiment']['output'] = "{}-{}".format(fname, str_o)
            conf['experiment']['agent']['communicator']['addr'] = "tcp://localhost:{}".format(port)
            conf['experiment']['test_agent']['communicator']['addr'] = "tcp://localhost:{}".format(port)
            conf['experiment']['load_file'] = "leo_drl_rbdl_balancing-{}-last".format(str_o)

            conf = remove_viz(conf)
            write_cfg(list_of_new_cfgs[-1], conf)

    print(list_of_new_cfgs)

    return list_of_new_cfgs, port


######################################################################################
def mp_run(cfg):
    # Multiple copies can be run on one computer at the same time, which results in the same seed for a random generator.
    # Thus we need to wait for a second or so between runs
    global counter
    global cores
    with counter_lock:
        wait = counter.value
        counter.value += 2
    sleep(wait)
    print('wait finished {0}'.format(wait))
    # Run the experiment
    start(cfg)

######################################################################################
def init(cnt, num):
    """ store the counter for later use """
    global counter
    global cores
    counter = cnt
    cores = num

######################################################################################
def do_multiprocessing_pool(args, list_of_new_cfgs):
    """Do multiprocesing"""
    counter = multiprocessing.Value('i', 0)
    cores = multiprocessing.Value('i', args.cores)
    print('cores {0}'.format(cores.value))
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(args.cores, initializer = init, initargs = (counter, cores))
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        pool.map(mp_run, list_of_new_cfgs)
    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
    pool.join()
######################################################################################

def read_cfg(cfg):
    """Read configuration file"""
    # check if file exists
    yfile = '../grl/qt-build/cfg/%s' % cfg
    if not os.path.isfile(yfile):
        print('File %s not found' % yfile)
        sys.exit()

    # open configuration
    stream = open(yfile, 'r')
    conf = yaml.load(stream)
    stream.close()
    return conf
######################################################################################

def write_cfg(outCfg, conf):
    """Write configuration file"""
    # create local yaml configuration file
    outfile = open(outCfg, 'w')
    yaml.dump(conf, outfile)
    outfile.close()
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
  return dumper.represent_dict(data.iteritems())
######################################################################################


def dict_constructor(loader, node):
  return collections.OrderedDict(loader.construct_pairs(node))
######################################################################################


def socket_free(host, port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
        if sock.connect_ex((host, port)) == 0:
            return 1
        else:
            return 0


def port_select(port):
    while True:
        port = port + 1
        if socket_free('localhost', port):
            return port


######################################################################################

if __name__ == "__main__":
    main()

