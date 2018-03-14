#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:53:20 2018

@author: ivan
"""
import numpy as np
import pickle
import pdb
import traceback

def _rejection_sampling(p, categories, n):
    x = []
    for i in range(n):
        while True:
            ci = np.random.randint(len(categories))
            if np.random.uniform() <= p[ci]:
                x.append(categories[ci])
                break
    return x


class opt_ce(object):
    def __init__(self, popsize, categories):
        self.popsize = popsize
        self.categories = categories
        self.psize = len(self.categories)
        self.pa = [0.5]*self.psize
        self.pa = [ x/sum(self.pa) for x in self.pa]
        self.pb = [0.5]*self.psize
        self.pb = [ x/sum(self.pb) for x in self.pb]
        self.alpha = 0.8
        self.Xi = []
        self.Yi = []

    def stop(self):
        return False

    def ask(self):
        # sample according to rare event probability distributions
        try:
            a = _rejection_sampling(self.pa, self.categories, self.popsize)
            b = _rejection_sampling(self.pb, self.categories, self.popsize)
        except KeyboardInterrupt:
            print('opt_ce: rejection_sampling KeyboardInterrupt')
            pdb.set_trace()
        sol = list(zip(a,b))
        return sol

    def tell(self, solutions, damage):
        # sort damage
        try:
            idxs =np.argsort(np.array(damage)) # minimum first
            quantile_idx = idxs[:int(len(damage)*0.1)]
            if len(quantile_idx) > 0: # if there are enough samples
                best = [solutions[x] for x in quantile_idx]
                besta, bestb = zip(*best)
                for i in range(self.psize):
                    cc = self.categories[i]
                    pa_hat = sum([1 for x in besta if x==cc]) / len(besta)
                    pb_hat = sum([1 for x in bestb if x==cc]) / len(bestb)
                    self.pa[i] = self.alpha * self.pa[i] + (1-self.alpha) * pa_hat
                    self.pb[i] = self.alpha * self.pb[i] + (1-self.alpha) * pb_hat
                self.pa = [ x/sum(self.pa) for x in self.pa]
                self.pb = [ x/sum(self.pb) for x in self.pb]
                self.Xi += solutions
                self.Yi += damage
            else:
                print('Not enough samples? Maybe damage is None everywhere.')
                pdb.set_trace()
        except Exception as e:
            print('Something went wrong in tell():\n{}\n{}'.format(e, traceback.format_exc()))
            pdb.set_trace()
        return quantile_idx

    def reeval(self, g, solutions, damage, hp):
        pass

    def log(self, root, alg, g, damage_info, reeval_damage_info, best):
        with open('{}/opt_ce.txt'.format(root), 'w') as f:
            for p in zip(self.Xi,self.Yi):
                f.write(str(p)+'\n')

        with open('{}/{}-g{:04}.txt'.format(root, alg, g), 'w') as f:
            for i in range(self.psize):
                f.write('{}'.format(self.categories[i]).rjust(6) + ': ' +
                        '{:0.3f}'.format(self.pa[i]).rjust(6)  +
                        '{:0.3f}'.format(self.pb[i]).rjust(6) + '\n')

            f.write('\n\n')
            for i, di in enumerate(damage_info):
                best_symb = '*' if i in best else ' '
                f.write('{:2d}'.format(i).rjust(3) + ': ' + best_symb +  str(di) + '\n')

    def save(self, root, fname):
        with open(root+'/'+fname, 'wb') as f:
            pickle.dump(self,f,2)

    @classmethod
    def load(cls, root, fname):
        with open(root+'/'+fname, 'rb') as f:
            return pickle.load(f)


