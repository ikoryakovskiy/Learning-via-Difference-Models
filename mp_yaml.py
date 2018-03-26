from __future__ import division
import multiprocessing
import os
import os.path
import yaml, collections, io
import sys
import itertools
import signal
import random
from datetime import datetime
import yaml, glob

from ddpg import parse_args, cfg_run

random.seed(datetime.now())

# Usage:
# options = [flatten(tupl) for tupl in options]
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def main():
    args = parse_args()

    if args['cores']:
        arg_cores = min(multiprocessing.cpu_count(), args['cores'])
    else:
        arg_cores = min(multiprocessing.cpu_count(), 32)
    print('Using {} cores.'.format(arg_cores))

    # Parameters
    L = []
    for f in glob.glob('cl/*-02_walking.yaml'):
        with open(f, 'r') as file:
            config = yaml.load(file)
        config['steps'] = config['steps'] + 200000
        L.append(config)

    len(L)

    random.shuffle(L)
    for config in L:
        a,b = config['output'].split('/')
        with open("{}/_{}.yaml".format(a,b), 'w', encoding='utf8') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

    #do_multiprocessing_pool(arg_cores, L)


######################################################################################
def mp_run(cfg):
    print('mp_run of {}'.format(cfg))
    # Read configuration
    try:
        file = open(cfg, 'r')
    except IOError:
        print("Could not read file: {}".format(cfg))
        sys.exit()
    with file:
        args = yaml.load(file)

    # Run the experiment
    try:
        cfg_run(**args)
    except Exception:
        print('mp_run {} failid to exit correctly'.format(cfg))
        sys.exit()


######################################################################################
def do_multiprocessing_pool(arg_cores, list_of_new_cfgs):
    """Do multiprocesing"""
    cores = multiprocessing.Value('i', arg_cores)
    print('cores {0}'.format(cores.value))
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(arg_cores)
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

