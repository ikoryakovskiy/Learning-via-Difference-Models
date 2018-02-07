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

    def __init__(self, input_dim, output_dim, config):
        self.i_dim = input_dim
        self.o_dim = output_dim
        self.learning_rate = config["cl_lr"]
        #self.l2 = config["cl_l2_reg"]
        self.layer_norm = config["cl_layer_norm"]
        self.w_num = 0          # total number of weights

        self.nn_i_dim = [self.i_dim]
        self.nn_activation = []
        self.nn_size = []
        for layer in config["cl_structure"].split(";"):
            activation, size = layer.split("_")
            self.w_num += self.nn_i_dim[-1]*int(size) + int(size)
            self.nn_activation.append(activation)
            self.nn_size.append(int(size))
            self.nn_i_dim.append(int(size)) # for the next layer

        self.norm_int = False
        if self.nn_activation[-1] == 'tanh':
            self.norm_int = True

        self.num_layers = len(self.nn_size)

        # Create the curriculum switching network
        self.inputs, self.out = self.create_curriculum_network()
        self.network_params = [v for v in tf.trainable_variables() if 'curriculum' in v.name]


    def create_curriculum_network(self):
        inputs = tflearn.input_data(shape=[None, self.i_dim])

        layer = inputs
        for i in range(self.num_layers):
            weights_init = tflearn.initializations.uniform(minval=-1/sqrt(self.nn_i_dim[i]), maxval=1/sqrt(self.nn_i_dim[i]))
            new_layer = tflearn.fully_connected(layer, self.nn_size[i], name="curriculumLayer{}".format(i), weights_init=weights_init)

            if self.layer_norm:
                new_layer = tflearn.layers.normalization.batch_normalization(new_layer, name="curriculumLayer{}_norm".format(i))

            if self.nn_activation[i] == 'relu':
                new_layer = tflearn.activations.relu(new_layer)
            elif self.nn_activation[i] == 'tanh':
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

    def train(self, sess, inputs, predicted_q_value):
        return sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.predicted_q_value: predicted_q_value
        })


    def predict(self, sess, inputs):
        r = sess.run(self.out, feed_dict={self.inputs: inputs})[0]

        modes = ['balancing', 'walking']
        if len(modes) == 2 and not self.norm_int:
            idx = int(r[0] > 0)
        else:
            eps = 1E-7
            bins = np.linspace(-1-eps, 1+eps, len(modes)+1) # forces NN output to be tanh
            idx = np.digitize(r, bins, right=True)[0] - 1 # index starts at 1, and include 0 as a left bin
        return modes[idx], r[0]

