from __future__ import division
import multiprocessing
import sys
import signal
import random
from datetime import datetime
import os
import cma
import pickle
import numpy as np

from ptracker import PerformanceTracker
from cl_main import cl_run
from ddpg import parse_args

random.seed(datetime.now())

def main():
    alg = 'ddpg'
    args = parse_args()

    if args['cores']:
        arg_cores = min(multiprocessing.cpu_count(), args['cores'])
    else:
        arg_cores = min(multiprocessing.cpu_count(), 32)
    print('Using {} cores.'.format(arg_cores))

    # important defaults
    args['cl_on'] = True
    args['rb_min_size'] = 500
    args['reach_reward'] = 1422.66
    args['steps'] = 300000
    popsize = None
    G = 1000


#    args['steps'] = 1000
#    args['seed']  = 0
#    popsize = 2
#    G = 1

    # Parameters
    starting_task = 'balancing'
    tasks = {'balancing': 'cfg/leo_balancing.yaml', 'walking': 'cfg/leo_walking.yaml'}

    root = "cl"
    if not os.path.exists(root):
        os.makedirs(root)

    # calulate number of weights
    pt = PerformanceTracker()
    input_dim = pt.get_v_size()
    w_num = 0
    fan_in = input_dim
    for layer in args["cl_structure"].split(";"):
        _, size = layer.split("_")
        w_num += fan_in*int(size) + int(size)
        fan_in = int(size)

    # initialize CMA-ES with all mean zeros
    cma_inopts = {}
    if args['seed'] == None:
        cma_inopts['seed'] = int.from_bytes(os.urandom(4), byteorder='big', signed=False) // 2
    else:
        cma_inopts['seed'] = args['seed'] + 1 # cma treats 0 as a random seed
    cma_inopts['popsize'] = popsize
    init = [0] * w_num
    es = cma.CMAEvolutionStrategy(init, 0.5, cma_inopts)

    g = 1
    while not es.stop() and g <= G:
        solutions = es.ask()

        # preparation
        mp_cfgs = []
        for run, solution in enumerate(solutions):
            cpy_args = args.copy()
            cpy_args['output']  = '{}/{}-g{:04}-mp{}'.format(root, alg, g, run)
            cpy_args['cl_save'] = '{}/{}-nn-g{:04}-mp{}'.format(root, alg, g, run)
            cpy_args['cl_load'] = '{}/{}-nn-g{:04}-mp{}'.format(root, alg, g-1, run)
            np.save(cpy_args['cl_load'], solution)
            mp_cfgs.append( (cpy_args, tasks, starting_task) )

        # evaluating
        if popsize != 2:
            damage_info = do_multiprocessing_pool(arg_cores, mp_cfgs)
            damage, info = zip(*damage_info)
        else:
            # for debug purpose
            config, tasks, starting_task = mp_cfgs[0]
            (damage0, cl_info0) = cl_run(tasks, starting_task, **config)
            config, tasks, starting_task = mp_cfgs[1]
            (damage1, cl_info1) = cl_run(tasks, starting_task, **config)
            damage = [damage0, damage1]
            info = [cl_info0, cl_info1]
            damage_info = zip(damage, info)

        # update cma
        es.tell(solutions, damage)
        es.logger.add()
        res = es.result_pretty()
        with open('{}/{}-g{:04}.txt'.format(root, alg, g), 'w') as f:
            f.write(str(res.fbest)+'\n')
            f.write(str(res.xbest)+'\n')
            f.write(str(res.xfavorite)+'\n')
            f.write(str(res.stds)+'\n\n')
            for di in damage_info:
                f.write(str(di)+'\n')

        with open('{}/cmaes.pkl'.format(root), 'wb') as output:
            pickle.dump(es, output, pickle.HIGHEST_PROTOCOL)

        # new iteration
        g += 1

    print(es.result_pretty())



######################################################################################
def mp_run(mp_cfg):
    config, tasks, starting_task = mp_cfg
    # Run the experiment
    try:
        return cl_run(tasks, starting_task, **config)
    except Exception:
        print('mp_run {} failid to exit correctly'.format(config['output']))
        sys.exit()


######################################################################################
def do_multiprocessing_pool(arg_cores, mp_cfgs):
    """Do multiprocesing"""
    cores = multiprocessing.Value('i', arg_cores)
    print('cores {0}'.format(cores.value))
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(arg_cores)
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        damage_info = pool.map(mp_run, mp_cfgs)
    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
    pool.join()
    return damage_info


######################################################################################
if __name__ == "__main__":
    main()

