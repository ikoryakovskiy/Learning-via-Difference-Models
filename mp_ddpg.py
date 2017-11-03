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

    prepare_multiprocessing()
    # for walking with yaml files
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
    yaml.add_representer(collections.OrderedDict, dict_representer)
    yaml.add_constructor(_mapping_tag, dict_constructor)

    # Parameters
    runs = range(3)
    gamma = [0.99, 0.97]
    actor_learning_rate = [0.00005, 0.0001, 0.001]
    critic_learning_rate = [0.0005, 0.001, 0.01]
    minibatch_size = [64, 128]

    options = []
    for r in itertools.product(gamma, actor_learning_rate, critic_learning_rate, minibatch_size, runs): options.append(r)
    options = [flatten(tupl) for tupl in options]

    configs = [
                "leo/drl/rbdl_ddpg.yaml"
              ]

    L = rl_run_param(args, configs, options)

    do_multiprocessing_pool(args, L)

######################################################################################
def rl_run_param(args, list_of_cfgs, options):
    list_of_new_cfgs = []

    loc = "tmp"
    if not os.path.exists(loc):
        os.makedirs(loc)

    port = 5557
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

            conf['experiment']['output'] = "{}-{}".format(fname, str_o)
            conf['experiment']['agent']['communicator']['addr'] = "tcp://localhost:{}".format(port)
            conf['experiment']['test_agent']['communicator']['addr'] = "tcp://localhost:{}".format(port)

            conf['experiment']['environment']['task']['gamma'] = o[0]
            conf['ddpg_param']['learning']['actor_learning_rate'] = o[1]
            conf['ddpg_param']['learning']['critic_learning_rate'] = o[2]
            conf['ddpg_param']['replay_buffer']['minibatch_size'] = o[3]
            conf['ddpg_param']['replay_buffer']['max_size'] = 1000000

            conf = remove_viz(conf)
            write_cfg(list_of_new_cfgs[-1], conf)
            port = port + 1

    print(list_of_new_cfgs)

    return list_of_new_cfgs

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

def prepare_multiprocessing():
    # clean bailing.out file
    f = open("bailing.out", "w")
    f.close()
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

if __name__ == "__main__":
    main()
