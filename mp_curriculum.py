from __future__ import division
import multiprocessing
import os
import os.path
import yaml, collections, io
from time import sleep
import itertools
import signal
import random
from datetime import datetime

from ddpg import parse_args, cfg_run

counter_lock = multiprocessing.Lock()
cores = 0
random.seed(datetime.now())

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def main():
    alg = 'ddpg'
    args = parse_args()

    if args['cores']:
        arg_cores = min(multiprocessing.cpu_count(), args['cores'])
    else:
        arg_cores = min(multiprocessing.cpu_count(), 32)
    print('Using {} cores.'.format(arg_cores))

    # Parameters
    runs = range(10)
    steps = [50]

    options = []
    for r in itertools.product(steps, runs): options.append(r)
    options = [flatten(tupl) for tupl in options]

    configs = {
                "balancing" : "cfg/rbdl_py_balancing.yaml",
              }
    L = rl_run(configs, alg, options)

    steps = [300]
    options = []
    for r in itertools.product(steps, runs): options.append(r)
    options = [flatten(tupl) for tupl in options]
    configs = {
                "walking" : "cfg/rbdl_py_walking.yaml",
                "curriculum" : "cfg/rbdl_py_walking.yaml",
              }
    L1 = rl_run(configs, alg, options)

    steps = [250]
    options = []
    for r in itertools.product(steps, runs): options.append(r)
    options = [flatten(tupl) for tupl in options]
    configs = {
                "walking_after_balancing" : "cfg/rbdl_py_walking.yaml",
                "curriculum_after_balancing" : "cfg/rbdl_py_walking.yaml",
              }
    L2 = rl_run(configs, alg, options, init = "ddpg-balancing-5000000")

    do_multiprocessing_pool(arg_cores, L)
    L3 = L1+L2
    random.shuffle(L3)
    do_multiprocessing_pool(arg_cores, L3)


######################################################################################
def rl_run(dict_of_cfgs, alg, options, init = ''):
    list_of_new_cfgs = []

    loc = "tmp"
    if not os.path.exists(loc):
        os.makedirs(loc)

    for key in dict_of_cfgs:
        args = parse_args()
        cfg = dict_of_cfgs[key]

        for o in options:
            str_o = "-".join(map(lambda x : "{:06d}".format(int(round(100000*x))), o[:-1]))  # last element in 'o' is reserved for mp
            if not str_o:
                str_o += "mp{}".format(o[-1])
            else:
                str_o += "-mp{}".format(o[-1])
            print("Generating parameters: {}".format(str_o))

            # create local filename
            list_of_new_cfgs.append( "{}/{}-{}-{}.yaml".format(loc, alg, key, str_o) )

            args['cfg'] = cfg
            args['steps'] = o[0]*1000
            args['save'] = True

            if 'curriculum' in key:
                args['curriculum'] = 'rwForward_50_300_10'

            if init:
                args['load_file'] = "{}-mp{}-last".format(init, o[-1])

            args['output'] = "{}-{}-{}".format(alg, key, str_o)

            with io.open(list_of_new_cfgs[-1], 'w', encoding='utf8') as file:
                yaml.dump(args, file, default_flow_style=False, allow_unicode=True)

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
    with open(cfg, 'r') as file:
        args = yaml.load(file)
    cfg_run(**args)


######################################################################################
def init(cnt, num):
    """ store the counter for later use """
    global counter
    global cores
    counter = cnt
    cores = num


######################################################################################
def do_multiprocessing_pool(arg_cores, list_of_new_cfgs):
    """Do multiprocesing"""
    counter = multiprocessing.Value('i', 0)
    cores = multiprocessing.Value('i', arg_cores)
    print('cores {0}'.format(cores.value))
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(arg_cores, initializer = init, initargs = (counter, cores))
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        pool.map(mp_run, list_of_new_cfgs)
    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
    pool.join()
######################################################################################


if __name__ == "__main__":
    main()

