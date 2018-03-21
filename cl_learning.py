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

from ptracker import PerformanceTracker
from cl_main import cl_run
from ddpg import parse_args

from opt_cmaes import opt_cmaes
from opt_bo import opt_bo
from spherical import cart2sph
from logger import Logger

random.seed(datetime.now())


class Helper(object):
    def __init__(self, base_cfg, root, alg, tasks, starting_task, arg_cores, use_mp=True):
        self.base_cfg = base_cfg
        self.root = root
        self.alg = alg
        self.tasks = tasks
        self.starting_task = starting_task
        self.arg_cores = arg_cores
        self.use_mp = use_mp
        self.reeval_damage_info = None

    def gen_base_(self, g, mp):
        cpy_cfg = self.base_cfg.copy()
        cpy_cfg['output']  = '{}/{}-g{:04}-mp{}'.format(self.root, self.alg, g, mp)
        cpy_cfg['cl_save'] = '{}/{}-nn-g{:04}-mp{}'.format(self.root, self.alg, g, mp)
        cpy_cfg['cl_load'] = '{}/{}-nn-g{:04}-mp{}'.format(self.root, self.alg, g-1, mp)
        if cpy_cfg['seed'] == None:
            cpy_cfg['seed'] = int.from_bytes(os.urandom(4), byteorder='big', signed=False) // 2
        return cpy_cfg

    def gen_cfg(self, solutions=None, g=1, begin=0):
        mp_cfgs = []
        for run, solution in enumerate(solutions):
            cfg = self.gen_base_(g, begin+run)
            if solution:
                np.save(cfg['cl_load'], solution)
            mp_cfgs.append( (cfg, self.tasks, self.starting_task) )
        return mp_cfgs

    def gen_cfg_steps(self, solutions=None, g=1, begin=0):
        mp_cfgs = []
        for run, solution in enumerate(solutions):
            cfg = self.gen_base_(g, begin+run)
            if solution:
                cfg['steps'] = solution
            mp_cfgs.append( (cfg, self.tasks, self.starting_task) )
        return mp_cfgs

    def run(self, mp_cfgs, reeval=False):
        if self.use_mp:
            damage_info = None
            while not damage_info: #protection from cases when something wrong happens with learning.
                damage_info = do_multiprocessing_pool(self.arg_cores, mp_cfgs)
                if not damage_info:
                    print('do_multiprocessing_pool returned None')
                    pdb.set_trace()
            damage, info, params = zip(*damage_info)
        else:
            # for debug purpose
            damage, info, params = [], [], []
            for cfg in mp_cfgs:
                config, tasks, starting_task = cfg
                (damage0, cl_info0, params0) = cl_run(tasks, starting_task, **config)
                damage.append(damage0)
                info.append(cl_info0)
                params.append(params0)
            damage_info = zip(damage, info, params)

        if not reeval:
            self.damage_info = list(damage_info).copy()
        else:
            self.reeval_damage_info = list(damage_info).copy()
        return damage


    def reeval_cfgs(self, solutions, g, begin):
        mp_cfgs = self.gen_cfg(solutions, g, begin)
        return mp_cfgs


def main():
    prepare_multiprocessing()
    alg = 'ddpg'
    args = parse_args()

    if args['cores']:
        arg_cores = min(multiprocessing.cpu_count(), args['cores'])
    else:
        arg_cores = min(multiprocessing.cpu_count(), 32)
    print('Using {} cores.'.format(arg_cores))

    # important defaults
    # 2-stage curriculum
#    args['cl_structure'] = 'cl:_1'
#    starting_task = 'balancing'

    # 3-stage curriculum
    args['cl_structure'] = 'cl:fc__2'
    starting_task = 'balancing_tf'

    args['mp_debug'] = True
    args['perf_td_error'] = True
    args['perf_l2_reg'] = True
    args['rb_min_size'] = 1000
    args['reach_return'] = 1422.66
    args['default_damage'] = 4035.00
    args['steps'] = 300000
    args['cl_depth'] = 1
    args['cl_l2_reg'] = 1000 # well-posing problem
    args['cl_cmaes_sigma0'] = 1.0
    popsize = 4
    resample = 4
    reeval_num0 = 5
    G = 500
    use_mp = True
    reeval = True

#    args['mp_debug'] = False
#    args['steps'] = 1500
#    popsize = 4
#    resample = 4
#    #reeval_num0 = 2
#    #args['seed']  = 1
#    G = 10
#    use_mp = False
#    reeval = False

    # Tasks
    tasks = {
            'balancing_tf': 'cfg/leo_balancing_tf.yaml',
            'balancing':    'cfg/leo_balancing.yaml',
            'walking':      'cfg/leo_walking.yaml'
            }

    root = "cl"
    if not os.path.exists(root):
        os.makedirs(root)

    # calulate number of weights
    pt = PerformanceTracker(depth=args['cl_depth'], input_norm=args["cl_running_norm"])
    input_dim = pt.get_v_size()
    w_num = 0
    fan_in = input_dim
    for layer in args["cl_structure"].split(";"):
        _, size = layer.split("_")
        w_num += fan_in*int(size) + int(size)
        fan_in = int(size)

    #opt = opt_cmaes(args, w_num, popsize, reeval_num0)
    search_space = (-1.0, 1.0)
    opt = opt_bo(args, w_num, popsize, resample, search_space)

    hp = Helper(args, root, alg, tasks, starting_task, arg_cores, use_mp=use_mp)

    g = 1
    while not opt.stop() and g <= G:
        if args['mp_debug']:
            sys.stdout = Logger(root + "/stdout-g{:04}.log".format(g))
            print("Should work")

        solutions = opt.ask()

        if args["cl_reparam"] == "spherical":
            resol = []
            for s in solutions:
                resol.append(cart2sph(s))
        elif args["cl_reparam"] == "cartesian":
            resol = solutions

        # preparation
        mp_cfgs = hp.gen_cfg(resol, g)

        # evaluating
        damage = hp.run(mp_cfgs)

        # update using *original* solutions
        _, rejected = opt.tell(solutions, damage)

        # reevaluation to prevent prepature convergence
        if reeval:
            opt.reeval(g, solutions, damage, hp)

        # logging
        opt.log(root, alg, g, hp.damage_info, hp.reeval_damage_info, rejected)
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

