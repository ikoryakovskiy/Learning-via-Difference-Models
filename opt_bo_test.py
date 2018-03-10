#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:47:11 2018

@author: ivan
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.space import Real
from skopt.utils import create_result
from skopt.plots import plot_objective, partial_dependence, plot_evaluations

with open('opt.pkl', 'rb') as f:
    optimizer = pickle.load(f)

res = create_result(optimizer.Xi, optimizer.yi, optimizer.space, optimizer.rng,
                             models=optimizer.models)
plot_evaluations(res)
plot_objective(res)#, levels=30, n_samples=500)

#f, ax = plt.subplots(1, 7, sharey=True)
#for plt_i, j in enumerate([0, 1, 2, 3, 4, 5, 7]):
#    xi, yi, zi = partial_dependence(res.space, res.models[-1], i=6, j=j)
#    ax[plt_i].contourf(xi, yi, zi, 10, alpha=.75, cmap='jet')
#plt.show()

