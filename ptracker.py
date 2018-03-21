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
import warnings

class PerformanceTracker(object):

    def __init__(self, cfg, input_norm=None, output_norm=None):
        self.depth = cfg['cl_depth']
        self.running_norm = cfg["cl_running_norm"]
        self.cl_pt_shape = cfg['cl_pt_shape']
        if self.cl_pt_shape == None:
            warnings.warn("PerformanceTracker will not be used")
        else:
            self.dim = self.cl_pt_shape[1]
            self.db = deque()
            self.count = 0
            self.flatten_shape = [-1] + [i for i in self.cl_pt_shape]
            if self.running_norm:
                self.input_rms = RunningMeanStd(self.dim)

            if input_norm:
                self.in_mean = input_norm[0][:self.dim]
                self.in_std  = input_norm[1][:self.dim]
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
        return self.cl_pt_shape


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
            return np.array(v).reshape(self.cl_pt_shape)
        elif self.in_mean is not None and self.in_std is not None:
            v = []
            for i in range(self.depth):
                v.append(self._normalize(self.db[i], self.in_mean, self.in_std))
            return np.reshape(v, self.flatten_shape)
        else:
            return np.array(self.db).reshape(self.cl_pt_shape)


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

