#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 2017

@author: Ivan Koryakovskiy
"""
from collections import deque
import numpy as np
from running_mean_std import RunningMeanStd


class PerformanceTracker(object):

    def __init__(self, depth=3, input_norm=True):
        self.dim = 3
        self.depth = depth
        self.db = deque()
        self.count = 0
        self.input_norm = input_norm
        if self.input_norm:
            self.input_rms = RunningMeanStd(self.dim)

        # fill in with zeros
        for i in range(self.depth):
            self.db.append(np.zeros((1, self.dim)))
            self.count += 1


    def get_v_size(self):
        return self.dim*self.depth


    def normalize(self, x):
        return (x - self.input_rms.mean) / self.input_rms.std


    def add(self, indicators):
        assert(len(indicators) == self.dim)

        indicators = np.reshape(indicators, (1, self.dim))
        self.db.popleft()
        self.db.append(indicators)

        if self.input_norm:
            self.input_rms.update(indicators)


    def flatten(self):
        if self.input_norm:
            # normalize whole db
            v = np.empty(self.get_v_size())
            for i in range(self.depth):
                v[i*self.dim:i*self.dim+self.dim] = np.clip(self.normalize(self.db[i]), -5, 5)
            return np.array(v).reshape((-1, self.get_v_size()))
        else:
            return np.array(self.db).reshape((-1, self.get_v_size()))

