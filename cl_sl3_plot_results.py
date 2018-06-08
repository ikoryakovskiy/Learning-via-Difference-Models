#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:10:33 2018

@author: ivan
"""
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/ivan/work/scripts/py')
from my_plot.plot import export_plot

plt.close('all')

subfigprop3 = {
    'figsize': (3.5,3.5),
    'dpi': 80,
    'facecolor': 'w',
    'edgecolor': 'k'
}

colors = ['#4355bf', '#51843b', '#ffb171']


with open('cl_sl3_results.pkl', 'rb') as handle:
    tpfn_test = pickle.load(handle)


sensitivity, specificity = [], []
for tpfn in tpfn_test:
    tp, fn, tn, fp = tpfn
    sensitivity_, specificity_ = [], []

    for tp_, fn_, tn_, fp_ in zip(tp, fn, tn, fp):
        sensitivity_.append(tp_/(tp_+fn_))
        specificity_.append(tn_/(tn_+fp_))
    sensitivity.append(sensitivity_)
    specificity.append(specificity_)

sensitivity = np.array(sensitivity)
specificity = np.array(specificity)

inan = np.isnan(specificity)
specificity[inan] = 0

fig, axarr = plt.subplots(2, sharex=True, **subfigprop3)

labels = ['ub balancing', 'wb balancing', 'walking']
xiter = [x*10 for x in range(sensitivity.shape[0])]
for i in range(sensitivity.shape[1]):
    axarr[0].plot(xiter, sensitivity[:,i], linestyle='-', color=colors[i], linewidth = 1.5)
    axarr[1].plot(xiter, specificity[:,i], linestyle='-', color=colors[i], label=labels[i], linewidth = 1.5)

labels = ['Sensitivity', 'Specificity']
for ax, label in zip(axarr, labels):
    ax.grid(True, color='lightgray', linestyle = '-', linewidth = 0.5)
    ax.get_xaxis().set_label_coords(  0.5, -0.32)
    ax.get_yaxis().set_label_coords(-0.19, 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel(label)

axarr[-1].set_xlabel('Traning iteration')
plt.subplots_adjust(left=0.2, bottom=0.17, right=0.97, top=0.97, wspace=0.0, hspace=0.25)

axarr[-1].legend(bbox_to_anchor=(0.5, 0.4), loc='center', ncol=1, columnspacing=1, facecolor='w', edgecolor='w')

export_plot('cl_sl3_results')

plt.show(block=True)