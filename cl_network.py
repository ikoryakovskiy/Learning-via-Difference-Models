#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 2017

@author: Ivan Koryakovskiy
"""

import tensorflow as tf
import tflearn
from os.path import exists
from math import sqrt
import numpy as np

class CurriculumNetwork(object):
    """
    Input to the network is the performance characteristics, output is the prediction switcher.
    """

    def __init__(self, input_dim, output_dim, config, cl_mode_init):
        self.i_dim = input_dim
        self.o_dim = output_dim
        self.learning_rate  = config["cl_lr"]
        self.batch_norm     = config["cl_batch_norm"]
        self.l2             = config["cl_l2_reg"]
        self.cl_on          = config['cl_on']
        self.cl_constraints = config["cl_constraints"]

        # learning modes
        if self.cl_on == 2:
            self.modes = ['balancing', 'walking']
        else:
            self.modes = ['balancing_tf', 'balancing', 'walking']

        if cl_mode_init:
            self.current_idx = self.modes.index(cl_mode_init)
        else:
            self.current_idx = 0

        self.layer_i_dim = [self.i_dim]
        self.layer_activation = []
        self.layer_size = []
        self.w_num = 0          # total number of weights in NN
        for layer in config["cl_structure"].split(";"):
            activation, size = layer.split("_")
            self.w_num += self.layer_i_dim[-1]*int(size) + int(size)
            self.layer_activation.append(activation)
            self.layer_size.append(int(size))
            self.layer_i_dim.append(int(size)) # for the next layer

        if self.layer_activation[-1] == 'tanh':
            self.norm_int = True
        else:
            self.norm_int = False

        if self.layer_activation[-1] == 'softmax':
            self.supervised_learning = True
        else:
            self.supervised_learning = False

        if not self.supervised_learning:
            assert(self.cl_on == self.layer_size[-1]+1)

        self.num_layers = len(self.layer_size)

        # Create the curriculum switching network
        self.inputs, self.out = self.create_curriculum_network()
        self.network_params = [v for v in tf.trainable_variables() if 'curriculum' in v.name]

        # supervised training
        if self.layer_activation[-1] == 'softmax':
            self.labels = tf.placeholder("float", [None, self.layer_size[-1]])
            var = tf.add_n([ tf.nn.l2_loss(v) for v in self.network_params if 'bias' not in v.name ]) * self.l2
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.labels) + var)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

    def create_curriculum_network(self):
        inputs = tflearn.input_data(shape=[None, self.i_dim])

        layer = inputs
        for i in range(self.num_layers):
            weights_init = tflearn.initializations.uniform(minval=-1/sqrt(self.layer_i_dim[i]), maxval=1/sqrt(self.layer_i_dim[i]))
            new_layer = tflearn.fully_connected(layer, self.layer_size[i], name="curriculumLayer{}".format(i), weights_init=weights_init)

            if self.batch_norm:
                new_layer = tflearn.layers.normalization.batch_normalization(new_layer, name="curriculumLayer{}_norm".format(i))

            if self.layer_activation[i] == 'relu':
                new_layer = tflearn.activations.relu(new_layer)
            elif self.layer_activation[i] == 'tanh':
                new_layer = tflearn.activations.tanh(new_layer)

            if i < self.num_layers-1:
                layer = new_layer

        return inputs, new_layer


    def load(self, sess, fname):
        if exists(fname+'.npy'):
            params = np.load(fname+'.npy').squeeze()
            self.set_params(sess, params)
        else:
            saver = tf.train.Saver(self.network_params)
            saver.restore(sess, fname)
        print("Loaded curriculum from {}".format(fname))
        return sess


    def save(self, sess, fname, global_step = None):
        saver = tf.train.Saver(self.network_params)
        saver.save(sess, "./" + fname, global_step)


    def set_params(self, sess, params):
        v_vars = self.network_params
        w = np.empty((self.w_num,))
        i = 0
        for v in v_vars:
            #print(v.eval())
            shape = [int(d) for d in v.shape]
            size = np.prod(shape)
            w = np.reshape(params[i: i+size], shape) # row major
            assign_op = v.assign(w)
            sess.run(assign_op)
            #print(v.eval())
            i += size
        return w

    def get_params(self, sess):
        w_vars = sess.run(self.network_params)
        w = np.empty((self.w_num,))
        i = 0
        for v in w_vars:
            w[i: i+v.size] = np.reshape(v, (v.size,)) # row major
            i += v.size
        #print(w)
        return w


    def train(self, sess, batch_x, batch_y):
        return sess.run([self.optimizer, self.cost], feed_dict={
                self.inputs: batch_x,
                self.labels: batch_y
               })


    def predict(self, sess, inputs):
        r = sess.run(self.out, feed_dict={self.inputs: inputs})[0]

        # supervised learning
        if self.cl_on == 2 and self.supervised_learning:
            return int(r[0] > r[1])

        # meta-heuristic
        if self.cl_on == 2 and not self.norm_int:
            idx = int(r[0] > 0)
        elif self.cl_on == 3 and not self.norm_int:
            idx = int(r[0] > 0) + 2*int(r[1] > 0)
        else:
            eps = 1E-7
            bins = np.linspace(-1-eps, 1+eps, len(self.modes)+1) # requires tanh output layer of NN
            idx = np.digitize(r, bins, right=True)[0] - 1 # index starts at 1, and include 0 as a left bin

        # do not allow to go backwards in the curriculum
        idx = min([len(self.modes)-1, idx])
        if self.cl_constraints == 'monotonic':
            idx = max([self.current_idx, idx])

        self.current_idx = idx
        return self.modes[idx], r

