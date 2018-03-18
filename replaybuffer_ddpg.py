#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

"""
"""
from collections import deque
import random
import numpy as np
import pickle
import os

class ReplayBuffer(object):
    def __init__(self, config, o_dims):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = config["rb_max_size"]
        self.replay_buffer_count = 0
        self.replay_buffer = deque()
        self.save_filename = config['rb_save_filename']
        self.load_filename = config['rb_load_filename']
        self.o_dims = o_dims

        print("Replay Buffer save = '{}', load = '{}'".format(
                self.save_filename, self.load_filename))


    def replay_buffer_add(self, s, a, r, t, s2):
        if s.size == self.o_dims:
            experience = (s, a, r, t, s2)
        else:
            experience = (s[0:self.o_dims], a, r, t, s2[0:self.o_dims],
                          s2[self.o_dims])

        if self.replay_buffer_count < self.buffer_size:
            self.replay_buffer.append(experience)
            self.replay_buffer_count += 1
        else:
            self.replay_buffer.popleft()
            self.replay_buffer.append(experience)

        return False

    def size(self):
        return self.replay_buffer_count

    def sample_batch(self, batch_size):
        if self.replay_buffer_count < batch_size:
            batch = random.sample(self.replay_buffer, self.replay_buffer_count)
        else:
            batch = random.sample(self.replay_buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.deque.clear()
        self.replay_buffer_count = 0

    def load(self):
        """ Load experiences """
        if self.load_filename and os.path.isfile(self.load_filename + '.db'):
            with open(self.load_filename + '.db', 'rb') as f:
                self.replay_buffer = pickle.load(f)
                f.close()
                self.replay_buffer_count += len(self.replay_buffer)

    def save(self):
        if self.save_filename:
            with open(self.save_filename + '.db', 'wb') as f:
                pickle.dump(self.replay_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)


