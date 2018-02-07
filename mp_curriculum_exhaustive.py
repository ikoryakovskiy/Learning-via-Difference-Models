from __future__ import division
import multiprocessing
import os
import os.path
import yaml, io
import traceback
import itertools
import signal
import random
from datetime import datetime

from ddpg import parse_args, cfg_run

random.seed(datetime.now())

def main():
    prepare_multiprocessing()
    alg = 'ddpg'
    args = parse_args()

    if args['cores']:
        arg_cores = min(multiprocessing.cpu_count(), args['cores'])
    else:
        arg_cores = min(multiprocessing.cpu_count(), 32)
    print('Using {} cores.'.format(arg_cores))

    # Parameters
    runs = range(10, 20)
    bsteps = range(10, 160, 10)

    options = []
    for r in itertools.product(bsteps, runs): options.append(r)

    configs = {
                "balancing" : "cfg/leo_balancing.yaml",
              }
    L0 = rl_run(configs, alg, options, rb_save=True)

    ## Zero-shot walking
    wsteps = [300 - s for s in bsteps]
    options = []
    for r in itertools.product(wsteps, runs): options.append(r)
    configs = {
                "walking_after_balancing" : "cfg/leo_walking.yaml",
              }
    L1 = rl_run(configs, alg, options, load_file="ddpg-balancing-{}-1010", rb_load="ddpg-balancing-{}-1010")

    # Execute learning

    # longer runs first
    L0 = L0[::-1]
    L1 = L1[::-1]
    #print(L0)
    #print(L1)
    do_multiprocessing_pool(arg_cores, L0)
    do_multiprocessing_pool(arg_cores, L1)

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
        args['rb_min_size'] = 1000

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
            str_steps = "{:06d}".format(int(round(100000*(300-o[0]))))
            args['save'] = save

            if 'curriculum' in key:
                args['curriculum'] = 'rwForward_50_300_10'

            if load_file:
                load_file_with_steps = load_file.format(str_steps)
                args['load_file'] = "{}-mp{}".format(load_file_with_steps, o[-1])

            args['output'] = "{}-{}-{}".format(alg, key, str_o)

            if rb_save:
                args['rb_save_filename'] = args['output']

            if rb_load:
                load_file_with_steps = rb_load.format(str_steps)
                args['rb_load_filename'] = "{}-mp{}".format(load_file_with_steps, o[-1])

            # Threads start at the same time, to prevent this we specify seed in the configuration
            args['seed'] = int.from_bytes(os.urandom(4), byteorder='big', signed=False) // 2
            with io.open(list_of_new_cfgs[-1], 'w', encoding='utf8') as file:
                yaml.dump(args, file, default_flow_style=False, allow_unicode=True)

    print(list_of_new_cfgs)
    return list_of_new_cfgs


######################################################################################
def mp_run(cfg):
    print('mp_run of {}'.format(cfg))
    bailing = None
    # Try to read configuration
    file = None
    try:
        file = open(cfg, 'r')
    except Exception as e:
        bailing = "mp_run {}:\n{}\n".format(cfg, traceback.format_exc())

    # Run the experiment
    if file:
        with file:
            args = yaml.load(file)
        try:
            cfg_run(**args)
        except Exception as e:
            bailing = "mp_run {}:\n{}\n".format(cfg, traceback.format_exc())

    # take care of fails
    if bailing:
        f = open("bailing.out", "a")
        try:
            f.write(bailing + "\n")
        finally:
            f.close()


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
        print('Finished tasks')
    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
        print('Closing complete')
    pool.join()
    print('Joining complete')


######################################################################################
def prepare_multiprocessing():
    # clean bailing.out file
    f = open("bailing.out", "w")
    f.close()


######################################################################################
def boolList2BinString(lst):
    return ''.join(['1' if x else '0' for x in lst])


######################################################################################
if __name__ == "__main__":
    main()

