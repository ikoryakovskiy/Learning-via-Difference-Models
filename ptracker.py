#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 2017

@author: Ivan Koryakovskiy
"""
from collections import deque
import numpy as np
import pickle
from running_mean_std import RunningMeanStd


class PerformanceTracker(object):

    def __init__(self, depth=3, running_norm=False, input_norm=None, output_norm=None, dim=3):
        self.dim = dim
        self.depth = depth
        self.db = deque()
        self.count = 0
        self.running_norm = running_norm
        if self.running_norm:
            self.input_rms = RunningMeanStd(self.dim)

        if input_norm:
            self.in_mean = input_norm[0][:dim]
            self.in_std  = input_norm[1][:dim]
        else:
            self.in_mean = None
            self.in_std  = None

        if output_norm:
            self.out_mean = output_norm[0]
            self.out_std  = output_norm[1]
        else:
            self.out_mean = None
            self.out_std  = None

        # fill in with zeros
        for i in range(self.depth):
            self.db.append(np.zeros((1, self.dim)))
            self.count += 1


    def get_v_size(self):
        return self.dim*self.depth


    def add(self, indicators):
        assert(len(indicators) == self.dim)

        indicators = np.reshape(indicators, (1, self.dim))
        self.db.popleft()
        self.db.append(indicators)

        if self.running_norm:
            self.input_rms.update(indicators)

    def denormalize(self, x):
        if self.running_norm:
            x = self._denormalize(x, self.input_rms.mean, self.input_rms.std)
        elif self.out_mean is not None and self.out_std is not None:
            x = self._denormalize(x, self.out_mean, self.out_std)
        return x

    def flatten(self):
        if self.running_norm:
            # normalize whole db
            v = np.empty(self.get_v_size())
            for i in range(self.depth):
                v[i*self.dim:i*self.dim+self.dim] = np.clip(self._normalize(self.db[i], self.input_rms.mean, self.input_rms.std), -5, 5)
            return np.array(v).reshape((-1, self.get_v_size()))
        elif self.in_mean is not None and self.in_std is not None:
            v = np.empty(self.get_v_size())
            for i in range(self.depth):
                v[i*self.dim:i*self.dim+self.dim] = self._normalize(self.db[i], self.in_mean, self.in_std)
            return np.array(v).reshape((-1, self.get_v_size()))
        else:
            return np.array(self.db).reshape((-1, self.get_v_size()))


    def save(self, filename):
        with open(filename,'wb') as f:
            pickle.dump((self.in_mean, self.in_std, self.out_mean, self.out_std), f)


    def load(self, filename):
        with open(filename,'rb') as f:
            self.in_mean, self.in_std, self.out_mean, self.out_std = pickle.load(f)


    def _normalize(self, x, mean, std):
        return (x - mean) / std


    def _denormalize(self, x, mean, std):
        return x*std + mean

