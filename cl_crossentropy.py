from __future__ import division
import multiprocessing
import traceback
import signal
import random
from datetime import datetime
import os
import numpy as np
import sys
import pdb
import time
from logger import Logger
import math

from ptracker import PerformanceTracker
from cl_main import cl_run
from ddpg import parse_args
from cl_learning import Helper

from opt_ce import opt_ce

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

    args['mp_debug'] = True
    args['perf_td_error'] = True
    args['perf_l2_reg'] = True
    args['rb_min_size'] = 1000
    steps       = 300000
    steps_ub    = 100000
    steps_delta =   5000
    popsize = 16*6
    G = 100
    use_mp = True

#    args['mp_debug'] = False
##    steps       = 3000
##    steps_ub    = 1000
##    steps_delta =  500
#    G = 10
#    use_mp = False

    # Tasks
    tasks = {
            'balancing_tf': 'cfg/leo_balancing_tf.yaml',
            'balancing':    'cfg/leo_balancing.yaml',
            'walking':      'cfg/leo_walking.yaml'
            }
    starting_task = 'balancing_tf'

    root = "cl"
    if not os.path.exists(root):
        os.makedirs(root)

    #opt = opt_ce.load(root, 'opt.pkl')

    categories = range(21) #list(range(0, steps_ub, steps_delta))
    opt = opt_ce(popsize, categories)

    hp = Helper(args, root, alg, tasks, starting_task, arg_cores, use_mp=use_mp)

    balancing_tf = np.array(categories)/max(categories)
    balancing_tf = [int(steps_ub*(math.exp(3*x)-1)/(math.exp(3)-1)) for x in balancing_tf]
    balancing = np.array(categories) * steps_delta

    g = 1
    while not opt.stop() and g <= G:
        if args['mp_debug']:
            sys.stdout = Logger(root + "/stdout-g{:04}.log".format(g))
            print("Should work")

        solutions = opt.ask()

        # remap solutions
        resol = []
        for s in solutions:
            a, b = s
            a = balancing_tf[a]
            b = balancing[b]
            if   a == 0 and b == 0:
                rs = (-1, -1, steps)
            elif a == 0 and b > 0:
                rs = (-1, b, steps-b)
            elif a > 0 and a < b:
                rs = (a, b-a, steps-b)
            elif a > 0 and a >= b:
                rs = (a, -1, steps-a)
            else:
                print('something bad happened')
                pdb.set_trace()
            resol.append(rs)

        # preparation
        mp_cfgs = hp.gen_cfg_steps(resol, g)

        # evaluating
        damage = hp.run(mp_cfgs)

        # update using *original* solutions
        best = opt.tell(solutions, damage)

        # logging
        opt.log(root, alg, g, hp.damage_info, hp.reeval_damage_info, best)
        opt.save(root, 'opt.pkl')

        # new iteration
        g += 1


######################################################################################
def mp_run(mp_run):
    config, tasks, starting_task = mp_run
    bailing = None
    time.sleep(3*random.random())
    # Run the experiment
    try:
        ret = cl_run(tasks, starting_task, **config)
        print('mp_run: ' +  config['output'] + ' returning ' + '{}'.format(ret))
        return ret
    except Exception:
        bailing = "mp_run {}:\n{}\n".format(config['output'], traceback.format_exc())

    print('mp_run: ' +  config['output'] + ' could not return correctly')

    # take care of fails
    if bailing:
        f = open("bailing.out", "a")
        try:
            f.write(bailing + "\n")
        finally:
            f.close()

    return (None, None, None)


######################################################################################
def do_multiprocessing_pool(arg_cores, mp_cfgs):
    """Do multiprocesing"""
    cores = multiprocessing.Value('i', arg_cores)
    print('cores {0}'.format(cores.value))
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(arg_cores)
    signal.signal(signal.SIGINT, original_sigint_handler)
    damage_info = None
    try:
        damage_info = pool.map(mp_run, mp_cfgs)
        print('Finished tasks')
    except KeyboardInterrupt:
        pool.terminate()
        print('Termination complete')
    else:
        pool.close()
        print('Closing complete')
    pool.join()
    print('Joining complete')

    # Protection against a list with all None
    if damage_info and all(di[0] is None for di in damage_info):
        damage_info = None
    return damage_info


######################################################################################
def prepare_multiprocessing():
    # clean bailing.out file
    f = open("bailing.out", "w")
    f.close()


######################################################################################
if __name__ == "__main__":
    main()

