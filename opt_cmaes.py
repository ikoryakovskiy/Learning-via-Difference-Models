#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:53:20 2018

@author: ivan
"""
import os
import cma
import numpy as np


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

class opt_cmaes(object):
    def __init__(self, config, w_num, popsize, reeval_num0):
        # initialize CMA-ES with all mean zeros
        cma_inopts = {}
        if config['seed'] == None:
            cma_inopts['seed'] = int.from_bytes(os.urandom(4), byteorder='big', signed=False) // 2
        else:
            cma_inopts['seed'] = config['seed'] + 1 # cma treats 0 as a random seed
        cma_inopts['popsize'] = popsize
        cma_inopts['bounds'] = [-1, 1]
        init = [0] * w_num
        self.es = cma.CMAEvolutionStrategy(init, sigma0=config['cl_cmaes_sigma0'], inopts=cma_inopts)
        self.nh = MyNoiseHandler(self.es.N, maxevals=[0, reeval_num0, 5.01], parallel=True, aggregate=np.mean)
        self.logger = cma.CMADataLogger().register(self.es)

    def stop(self):
        return self.es.stop()

    def ask(self):
        return self.es.ask()

    def tell(self, solutions, damage):
        return self.es.tell(solutions, damage)

    def reeval(self, g, solutions, damage, hp):
        self.es.sigma *= self.nh(solutions, damage, hp, self.es.ask, args=(g, len(solutions)))  # see method __call__
        self.es.countevals += self.nh.evaluations_just_done

    def log(self, root, alg, g, damage_info, reeval_damage_info):
        self.logger.add(more_data = [self.nh.evaluations, self.nh.noiseS])
        res = self.es.result_pretty()

        with open('{}/{}-g{:04}.txt'.format(root, alg, g), 'w') as f:
            f.write(str(res.fbest)+'\n')
            f.write(str(res.xbest)+'\n')
            f.write(str(res.xfavorite)+'\n')
            f.write(str(res.stds)+'\n\n')

            for i, di in enumerate(damage_info):
                f.write('{:2d}'.format(i).rjust(3) + ': ' +  str(di) + '\n')

            if self.nh and reeval_damage_info:
                f.write('\n')
                for i, di in enumerate(reeval_damage_info):
                    f.write('{:2d}'.format(self.nh.mp_idxs[i]).rjust(3) + ': ' +  str(di) + '\n')
                f.write('\n' + self.nh.print() + '\n')