#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

"""
"""
from collections import deque
import random
import numpy as np
import pickle

class ReplayBuffer(object):
    def __init__(self, config):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = config["rb_max_size"]
        self.buffer_size_file = 0
        self.replay_buffer_count = 0
        self.replay_buffer = deque()
        self.save_filename = config['rb_save_filename']
        self.load_filename = config['rb_load_filename']
        self.buffer_size_file = config['rb_max_size']

        print("Replay Buffer save = '{}', load = '{}'".format(
                self.save_filename, self.load_filename))

        # load buffer now
        if self.load_filename:
            with open(self.load_filename) as f:
                self.replay_buffer = pickle.load(f)
                f.close()
                self.replay_buffer_count += len(self.replay_buffer)


    def replay_buffer_add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)

        if self.replay_buffer_count < self.buffer_size:
            self.replay_buffer.append(experience)
            self.replay_buffer_count += 1
        else:
            self.replay_buffer.popleft()
            self.replay_buffer.append(experience)

        '''
        if self.replay_buffer_count == self.buffer_size_file:
            if self.buffer_save:
                with open(self.save_filename, 'w') as f:
                    pickle.dump(self.replay_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        '''
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
