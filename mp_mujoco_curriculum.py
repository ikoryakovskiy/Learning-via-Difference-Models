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

# Usage:
# options = [flatten(tupl) for tupl in options]
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
    runs = range(4)
    reassess_for = ['']

    #####
    ## Zero-shot balancing HalfCheetah
    options = []
    steps = [200]
    for r in itertools.product(steps, reassess_for, runs): options.append(r)

    configs = {
                "HalfCheetah_balancing" : "RoboschoolHalfCheetahBalancing-v1",
              }
    L0_H = rl_run(configs, alg, options, rb_save=True)

    ## Zero-shot balancing Walker2d
    options = []
    steps = [300]
    for r in itertools.product(steps, reassess_for, runs): options.append(r)

    configs = {
                "Walker2d_balancing" : "RoboschoolWalker2dBalancing-v1",
              }
    L0_W = rl_run(configs, alg, options, rb_save=True)
    #####

    #####
    ## Zero-shot walking HalfCheetah & Walker2d
    steps = [1000]
    options = []
    for r in itertools.product(steps, reassess_for, runs): options.append(r)
    configs = {
                "HalfCheetah_walking" : "RoboschoolHalfCheetah-v1",
                "Walker2d_walking" : "RoboschoolWalker2d-v1",
              }
    L1 = rl_run(configs, alg, options)
    #####

    #####
    ## Only neural network without replay buffer HalfCheetah
    steps = [800]
    options = []
    for r in itertools.product(steps, reassess_for, runs): options.append(r)
    configs = {
                "HalfCheetah_walking_after_balancing" : "RoboschoolHalfCheetah-v1",
              }
    L2_H = rl_run(configs, alg, options, load_file="ddpg-HalfCheetah_balancing-20000000-1010")

    ## Only neural network without replay buffer Walker2d
    steps = [700]
    options = []
    for r in itertools.product(steps, reassess_for, runs): options.append(r)
    configs = {
                "Walker2d_walking_after_balancing" : "RoboschoolWalker2d-v1",
              }
    L2_W = rl_run(configs, alg, options, load_file="ddpg-Walker2d_balancing-30000000-1010")
    ####

    ####
    ## Replay buffer HalfCheetah
    steps = [800]
    reassess_for = ['walking_1_0', '']
    options = []
    for r in itertools.product(steps, reassess_for, runs): options.append(r)
    configs = {
                "HalfCheetah_walking_after_balancing" : "RoboschoolHalfCheetah-v1",
              }
    L3_H = rl_run(configs, alg, options, rb_load="ddpg-HalfCheetah_balancing-20000000-1010")
    L4_H = rl_run(configs, alg, options, load_file="ddpg-HalfCheetah_balancing-20000000-1010", rb_load="ddpg-HalfCheetah_balancing-20000000-1010")

    ## Replay buffer Walker2d
    steps = [700]
    reassess_for = ['walking_1_0', '']
    options = []
    for r in itertools.product(steps, reassess_for, runs): options.append(r)
    configs = {
                "Walker2d_walking_after_balancing" : "RoboschoolWalker2d-v1",
              }
    L3_W = rl_run(configs, alg, options, rb_load="ddpg-Walker2d_balancing-30000000-1010")
    L4_W = rl_run(configs, alg, options, load_file="ddpg-Walker2d_balancing-30000000-1010", rb_load="ddpg-Walker2d_balancing-30000000-1010")

    ####
    do_multiprocessing_pool(arg_cores, L0_H+L0_W)
    L = L1 + L2_H + L2_W + L3_H + L3_W + L4_H + L4_W
    random.shuffle(L)
    do_multiprocessing_pool(arg_cores, L)

######################################################################################
def opt_to_str(opt):
    str_o = ''
    for  o in opt[:-1]:  # last element in 'o' is reserved for mp
        try:
            fl = float(o) # converts to float numbers and bools
            str_o += "-{:06d}".format(int(round(100000*fl)))
        except ValueError:
            if o: # skip empty elements, e.g. ""
                str_o +='-' + o
    if str_o:
        str_o = str_o[1:]
    return str_o

######################################################################################
def rl_run(dict_of_cfgs, alg, options, save=True, load_file='', rb_save=False, rb_load=''):
    list_of_new_cfgs = []

    loc = "tmp"
    if not os.path.exists(loc):
        os.makedirs(loc)

    for key in dict_of_cfgs:
        args = parse_args()
        cfg = dict_of_cfgs[key]

        for o in options:
            str_o = opt_to_str(o)
            str_o += '-' + boolList2BinString([save, bool(load_file), rb_save, bool(rb_load)])
            if not str_o:
                str_o += "mp{}".format(o[-1])
            else:
                str_o += "-mp{}".format(o[-1])
            print("Generating parameters: {}".format(str_o))

            # create local filename
            list_of_new_cfgs.append( "{}/{}-{}-{}.yaml".format(loc, alg, key, str_o) )

            args['cfg'] = cfg
            args['steps'] = o[0]*1000
            args['rb_max_size'] = args['steps']
            args['reassess_for'] = o[1]
            args['save'] = save

            if 'curriculum' in key:
                args['curriculum'] = 'rwForward_50_300_10'

            if load_file:
                args['load_file'] = "{}-mp{}".format(load_file, o[-1])

            args['output'] = "{}-{}-{}".format(alg, key, str_o)

            if rb_save:
                args['rb_save_filename'] = args['output']

            if rb_load:
                args['rb_load_filename'] = "{}-mp{}".format(rb_load, o[-1])

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
def boolList2BinString(lst):
    return ''.join(['1' if x else '0' for x in lst])


######################################################################################
if __name__ == "__main__":
    main()
