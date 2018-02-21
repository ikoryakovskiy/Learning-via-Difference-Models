from __future__ import division
import multiprocessing
import traceback
import signal
import random
from datetime import datetime
import os
import cma
import numpy as np
import sys
import pdb

from ptracker import PerformanceTracker
from cl_main import cl_run
from ddpg import parse_args
from logger import Logger

random.seed(datetime.now())

class MyNoiseHandler(cma.NoiseHandler):
    def reeval(self, X, fit, helper, ask, args=()):
        """store two fitness lists, `fit` and ``fitre`` reevaluating some
        solutions in `X`.
        ``self.evaluations`` evaluations are done for each reevaluated
        fitness value.
        See `__call__`, where `reeval` is called.

        """
        self.fit = list(fit)
        self.fitre = list(fit)
        self.idx = self.indices(fit)
        if not len(self.idx):
            return self.idx
        evals = int(self.evaluations) if self.f_aggregate else 1
        fagg = np.median if self.f_aggregate is None else self.f_aggregate
        mp_cfgs = []
        self.mp_idxs = []
        g = args[0]
        begin = args[1]

        for i in self.idx:
            X_i = X[i]
            if self.epsilon:
                if self.parallel:
                    cfgs = helper.reeval_cfgs(ask(evals, X_i, self.epsilon), g, begin)
                    for j in range(len(cfgs)):
                        mp_cfgs.append(cfgs[j])
                        self.mp_idxs.append(i)
                    begin += len(cfgs)
                else:
                    raise Exception('This value of self.parallel is not supported')
            else:
                raise Exception('This value of self.epsilon is not supported')

        # resample function
        damage = helper.run(mp_cfgs, reeval=True) # damage follows the order of mp_cfgs and mp_idxs

        # calculate median value for all
        for i in self.idx:
            damage_of_index = [damage[j] for j, x in enumerate(self.mp_idxs) if x == i]
            self.fitre[i] = fagg(damage_of_index)

        self.evaluations_just_done = evals * len(self.idx)
        return self.fit, self.fitre, self.idx

    def print(self):
        return 'alphasigma = {:0.5f}, evaluations = {:0.5f}'.format(self.alphasigma, self.evaluations)


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

    def gen_cfg(self, solutions, g, begin=0):
        mp_cfgs = []
        for run, solution in enumerate(solutions):
            cpy_cfg = self.base_cfg.copy()
            cpy_cfg['output']  = '{}/{}-g{:04}-mp{}'.format(self.root, self.alg, g, begin+run)
            cpy_cfg['cl_save'] = '{}/{}-nn-g{:04}-mp{}'.format(self.root, self.alg, g, begin+run)
            cpy_cfg['cl_load'] = '{}/{}-nn-g{:04}-mp{}'.format(self.root, self.alg, g-1, begin+run)
            np.save(cpy_cfg['cl_load'], solution)
            mp_cfgs.append( (cpy_cfg, self.tasks, self.starting_task) )
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


    def log(self, g, cma_res, nh = None):
        with open('{}/{}-g{:04}.txt'.format(self.root, self.alg, g), 'w') as f:
            f.write(str(cma_res.fbest)+'\n')
            f.write(str(cma_res.xbest)+'\n')
            f.write(str(cma_res.xfavorite)+'\n')
            f.write(str(cma_res.stds)+'\n\n')

            for i, di in enumerate(self.damage_info):
                f.write('{:2d}'.format(i).rjust(3) + ': ' +  str(di) + '\n')

            if nh and self.reeval_damage_info:
                f.write('\n')
                for i, di in enumerate(self.reeval_damage_info):
                    f.write('{:2d}'.format(nh.mp_idxs[i]).rjust(3) + ': ' +  str(di) + '\n')
                f.write('\n' + nh.print() + '\n')


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
    args['mp_debug'] = True
    args['cl_on'] = 2
    args['perf_td_error'] = True
    args['perf_l2_reg'] = True
    args['rb_min_size'] = 1000
    args['reach_return'] = 1422.66
    args['default_damage'] = 4035.00
    args['steps'] = 300000
    args['cl_depth'] = 1
    args['cl_structure'] = '_1'
    args['cl_l2_reg'] = 1000 # well-posing problem
    args['cl_cmaes_sigma0'] = 1.0
    popsize = 15 # None
    reeval_num0 = 5
    G = 250
    use_mp = True
    reeval = True

#    args['steps'] = 1500
#    popsize = 2
#    reeval_num0 = 0
#    #args['seed']  = 123
#    G = 3
#    #use_mp = False
#    reeval = False

    # Parameters
    starting_task = 'balancing'
    tasks = {'balancing': 'cfg/leo_balancing.yaml', 'walking': 'cfg/leo_walking.yaml'}

    root = "cl"
    if not os.path.exists(root):
        os.makedirs(root)

    # calulate number of weights
    pt = PerformanceTracker(depth=args['cl_depth'], input_norm=args["cl_input_norm"])
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
    cma_inopts['bounds'] = [-1, 1]
    init = [0] * w_num
    es = cma.CMAEvolutionStrategy(init, sigma0=args['cl_cmaes_sigma0'], inopts=cma_inopts)
    nh = MyNoiseHandler(es.N, maxevals=[0, reeval_num0, 5.01], parallel=True, aggregate=np.mean)

    logger = cma.CMADataLogger().register(es)

    hp = Helper(args, root, alg, tasks, starting_task, arg_cores, use_mp=use_mp)

    g = 1
    while not es.stop() and g <= G:
        if args['mp_debug']:
            sys.stdout = Logger(root + "/stdout-g{:04}.log".format(g))
            print("Should work")

        solutions = es.ask()

        # preparation
        mp_cfgs = hp.gen_cfg(solutions, g)

        # evaluating
        damage = hp.run(mp_cfgs)

        # update cma
        es.tell(solutions, damage)

        # reevaluation to prevent prepature convergence
        if reeval:
            es.sigma *= nh(solutions, damage, hp, es.ask, args=(g, len(solutions)))  # see method __call__
            es.countevals += nh.evaluations_just_done

        # logging
        logger.add(more_data = [nh.evaluations, nh.noiseS])
        cma_res = es.result_pretty()
        hp.log(g, cma_res, nh)

        # new iteration
        g += 1

    print(es.result_pretty())


######################################################################################
def mp_run(mp_cfg):
    config, tasks, starting_task = mp_cfg
    bailing = None
    # Run the experiment
    try:
        ret = cl_run(tasks, starting_task, **config)
        print('mp_run: ' +  config['output'] + ' returning ' + '{}'.format(ret))
        return ret
    except Exception as e:
        bailing = "mp_run {}:\n{}\n".format(config['output'], traceback.format_exc())

    print('mp_run: ' +  config['output'] + ' could not return correctly')

    # take care of fails
    if bailing:
        f = open("bailing.out", "a")
        try:
            f.write(bailing + "\n")
        finally:
            f.close()

    return None

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

    # Protection against a list with any None => treat whole list as none
    if damage_info and None in damage_info:
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

