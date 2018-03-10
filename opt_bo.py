#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:53:20 2018

@author: ivan
"""
import numpy as np
import pickle

from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.space import Real
from outliers import dixon_test

class opt_bo(object):
    def __init__(self, config, w_num, popsize, resample, search_space):
        self.popsize = popsize
        self.resample = resample
        self.optimizer = Optimizer(
            dimensions=[Real(search_space[0], search_space[1])] * w_num,
            random_state=1, # use the same seed for repeatability
            n_initial_points=self.popsize*self.resample, # if self.resample > 1, then we will continue ask-tell cycles self.resample times
            acq_optimizer_kwargs = {'n_points':10000} # 'n_jobs' slows down, 'noise' seem not to be used
        )

    def stop(self):
        return False

    def ask(self):
        x = self.optimizer.ask(n_points=self.popsize)
        x = [xx for xx in x for i in range(self.resample)]
        return x

    def tell(self, solutions, damage):
        solutions = solutions[::self.resample]
        damage = chunks_(damage, self.resample)
        X, Y, rejected = [], [], []
        for batch_id, batch in enumerate(damage):
            valid_batch = [x for x in batch if x is not None]
            valid_batch, outliers = dixon_test(valid_batch, pres=-1)
            if len(valid_batch) > 0:
                batch_mean = np.mean(valid_batch)
                X.append(solutions[batch_id])
                Y.append(batch_mean)
            # else if all batch damages are None then we hope tell() will not
            # compalin about missing sample (according to manual, should be ok)
            if outliers:
                outliers = [x + self.resample*batch_id for x in outliers]
                rejected.append(outliers)

        res = self.optimizer.tell(X, Y)
        rejected = [y for x in rejected for y in x] # flatten list of rejected episodes' indexes
        return res, rejected

    def reeval(self, g, solutions, damage, hp):
        pass

    def log(self, root, alg, g, damage_info, reeval_damage_info, rejected):
        ibest = np.argmin(self.optimizer.yi)

        with open('{}/opt_bo.txt'.format(root), 'w') as f:
            for p in zip(self.optimizer.Xi,self.optimizer.yi):
                f.write(str(p)+'\n')

        with open('{}/{}-g{:04}.txt'.format(root, alg, g), 'w') as f:
            f.write(str(self.optimizer.yi[ibest])+'\n')
            f.write(str(self.optimizer.Xi[ibest])+'\n\n\n')

            for i, di in enumerate(damage_info):
                rej_symb = '*' if i in rejected else ' '
                f.write('{:2d}'.format(i).rjust(3) + ': ' + rej_symb +  str(di) + '\n')

    def save(self, root, fname):
        with open(root+'/'+fname, 'wb') as f:
            pickle.dump(self.optimizer, f)

    def load(self, root, fname):
        with open(root+'/'+fname, 'rb') as f:
            self.optimizer = pickle.load(f)

def chunks_(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))
