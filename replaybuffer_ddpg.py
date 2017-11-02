#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

"""
"""
from collections import deque
import random
import numpy as np
import yaml
import os
import pickle
from train_model import prediction

class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed, diff_sess = None, ddpg_cfg = 'config.yaml'):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.buffer_size_file = 0
        self.transitions_size_file = 0
        self.transitions_count = 0
        self.replay_buffer_count = 0
        self.transitions_buffer_count = 0
        self.replay_buffer = deque()
        self.transitions_buffer = deque()
        self.transitions = deque()
        self.transitions_save = 0
        self.transitions_load = 0
        self.buffer_save = 0
        self.buffer_load = 0
        self.diff = 0
        self.save_filename = None
        self.load_filename = None
        self.model_filename = None
        self.sess = diff_sess
        random.seed(random_seed)
        self.read_cfg(ddpg_cfg)

    def replay_buffer_add(self, s, a, r, t, s2, sd):
        experience = (s, a, r, t, s2, sd)

        if self.replay_buffer_count < self.buffer_size:
            self.replay_buffer.append(experience)
            self.replay_buffer_count += 1
            # print ("Buffer count:", self.buffer_count)
        else:
            self.replay_buffer.popleft()
            self.replay_buffer.append(experience)

        if self.replay_buffer_count == self.buffer_size_file:
            if self.buffer_save:
                with open(self.save_filename, 'w') as f:
                    pickle.dump(self.replay_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        return False

    def transitions_buffer_add(self, s, a, r, t, s2, sd):

        if self.transitions_save:
            experience = (s, a, r, t, s2, sd)
            self.transitions_buffer.append(experience)
            self.transitions_buffer_count +=1
            if self.transitions_buffer_count == self.transitions_size_file:
                with open(self.save_filename, 'w') as f:
                    pickle.dump(self.transitions_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print (len(self.transitions_buffer))
                return True
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
        self.transitions_count = 0
        self.replay_buffer_count = 0

    def read_cfg(self, cfg):
        path = os.path.dirname(os.path.abspath(__file__))
        yfile = '{}/{}'.format(path,cfg)
        print ("Loading Transitions and Replay Buffer from", yfile)
        if not os.path.isfile(yfile):
            print ("File %s not found" % yfile)
        else:
            # open configuration
            stream = open(yfile, 'r')
            conf = yaml.load(stream)
            if 'transitions' in conf:
                self.transitions_save = int(conf['transitions']['save'])
                self.transitions_load = int(conf['transitions']['load'])
                self.transitions_size_file = conf['transitions']['buffer_size']
            if 'replay_buffer' in conf:
                self.buffer_save = int(conf['replay_buffer']['save'])
                self.buffer_load = int(conf['replay_buffer']['load'])
                self.buffer_size_file = conf['replay_buffer']['buffer_size']

            self.diff = int(conf['difference_model'])
            print("Transitions save = {}, load = {}, diff = {}".format(self.transitions_save, self.transitions_load, self.diff))
            print("Replay Buffer save = {}, load = {}, diff = {}".format(self.buffer_save, self.buffer_load, self.diff))

            if self.transitions_save == 1:
                self.save_filename = conf['transitions']['save_filename']
            elif self.buffer_save == 1:
                self.save_filename = conf['replay_buffer']['save_filename']

            if self.transitions_load == 1:
                self.load_filename = conf['transitions']['load_filename']
                with open(self.load_filename) as f:
                    self.transitions = pickle.load(f)
                    print (len(self.transitions))
                    f.close()

            if self.buffer_load == 1:
                self.load_filename = conf['replay_buffer']['load_filename']
                with open(self.load_filename) as f:
                    self.replay_buffer = pickle.load(f)
                    f.close()
                    # self.update_replay_buffer()
                    self.replay_buffer_count += len(self.replay_buffer)

            if self.diff == 1:
                self.model_filename = conf['difference_model']['model_filename']
            stream.close()

    def sample_state_action(self, state, action, test, episode_start):

        if self.transitions_load == 1:
            if not episode_start:
                self.transitions_count += 1
            temp = self.transitions[self.transitions_count]
            state = temp[0]
            action = temp[1]
            return state, action

        else:
            return state, action

    def update_replay_buffer(self):

        s_batch = np.array([_[0] for _ in self.replay_buffer])
        a_batch = np.array([_[1] for _ in self.replay_buffer])
        r_batch = np.array([_[2] for _ in self.replay_buffer])
        t_batch = np.array([_[3] for _ in self.replay_buffer])
        s2_batch = np.array([_[4] for _ in self.replay_buffer])
        diff_state_old = np.array([_[5] for _ in self.replay_buffer])

        input = np.concatenate((s_batch, a_batch, s2_batch-diff_state_old), axis=1)
        s2_new = prediction(self.sess, input, 24, 18)
        diff_state_new = s2_new - (s2_batch - diff_state_old)
        self.replay_buffer.clear()

        for i in range(s_batch):
            self.replay_buffer.append((s_batch[i], a_batch[i], r_batch[i], t_batch[i], s2_new[i], diff_state_new[i]))





