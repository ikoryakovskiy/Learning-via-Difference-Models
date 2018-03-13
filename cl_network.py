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

###############################################################################
class FeedForwardNetwork(object):
    def __init__(self, input_dim, config):
        self.i_dim = input_dim
        self.learning_rate = config["cl_lr"]
        self.batch_norm = config["cl_batch_norm"]
        self.structure = config["cl_structure"]
        self._decode()
        self.inputs, self.out = self._create()
        self.network_params = [v for v in tf.trainable_variables() if 'curriculum' in v.name]


    def load(self, sess, fname):
        if exists(fname+'.npy'):
            params = np.load(fname+'.npy').squeeze()
            self._set_params(sess, params)
        else:
            saver = tf.train.Saver(self.network_params)
            saver.restore(sess, fname)
        print("Loaded curriculum from {}".format(fname))
        return sess


    def save(self, sess, fname, global_step = None):
        saver = tf.train.Saver(self.network_params)
        saver.save(sess, "./" + fname, global_step)


    def _decode(self):
        self.layer_size = [self.i_dim] # includes input layer
        self.layer_activation = []
        self.w_num = 0 # total number of weights in NN
        network_description = self.structure.split(":")[1].split(";")
        for layer in network_description:
            activation, size = layer.split("_")
            self.w_num += self.layer_size[-1]*int(size) + int(size)
            self.layer_activation.append(activation)
            self.layer_size.append(int(size))
        self.num_hidden_layers = len(self.layer_size)-1


    def _create(self):
        inputs = tflearn.input_data(shape=[None, self.i_dim])
        layer = inputs
        for i in range(self.num_hidden_layers):
            weights_init = tflearn.initializations.uniform(minval=-1/sqrt(self.layer_size[i]), maxval=1/sqrt(self.layer_size[i]))
            new_layer = tflearn.fully_connected(layer, self.layer_size[i+1], name="curriculumLayer{}".format(i), weights_init=weights_init)

            if self.batch_norm:
                new_layer = tflearn.layers.normalization.batch_normalization(new_layer, name="curriculumLayer{}_norm".format(i))

            if self.layer_activation[i] == 'relu':
                new_layer = tflearn.activations.relu(new_layer)
            elif self.layer_activation[i] == 'tanh':
                new_layer = tflearn.activations.tanh(new_layer)

            if i < self.num_hidden_layers-1:
                layer = new_layer
        return inputs, new_layer


    def _set_params(self, sess, params):
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


    def _get_params(self, sess):
        w_vars = sess.run(self.network_params)
        w = np.empty((self.w_num,))
        i = 0
        for v in w_vars:
            w[i: i+v.size] = np.reshape(v, (v.size,)) # row major
            i += v.size
        #print(w)
        return w


###############################################################################
class RecurrentNeuralNetwork(object):
    def __init__(self, input_dim, output_dim, config, cl_mode_init):
        pass

###############################################################################
class FeedForwardCurriculumNetwork(FeedForwardNetwork):
    def __init__(self, input_dim, config):
        super().__init__(input_dim, config)


    def predict(self, sess, inputs, num_stages):
        rr = sess.run(self.out, feed_dict={self.inputs: inputs})[0]

        if self.layer_activation[-1] == 'tanh':
            eps = 1E-7
            bins = np.linspace(-1-eps, 1+eps, num_stages+1) # requires tanh output layer of NN
            idx = np.digitize(rr, bins, right=True)[0] - 1 # index starts at 1, and include 0 as a left bin
        else:
            reversed_rr = rr[::-1]
            binary = map(int, reversed_rr > 0)
            binary_string = ''.join(map(str, binary))
            idx = int(binary_string, 2)

        return idx, rr


    def train(self, sess, batch_x, batch_y):
        # curriculum network is trained by metaheuristic
        pass


    def validate(self, num_stages):
        assert((self.layer_size[-1] == 1 and num_stages == 2) or # 2-stage curriculum
               (self.layer_size[-1] == 2 and num_stages == 3))   # 3-stage curriculum



###############################################################################
class FeedForwardSupervisedClassificationNetwork(FeedForwardNetwork):
    def __init__(self, input_dim, config):
        super().__init__(input_dim, config)
        self.l2 = config["cl_l2_reg"]
        self.learning_rate = config["cl_lr"]

        self.labels = tf.placeholder("float", [None, self.layer_size[-1]])
        var = tf.add_n([ tf.nn.l2_loss(v) for v in self.network_params if 'bias' not in v.name ]) * self.l2
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.labels) + var)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)


    def predict(self, sess, inputs):
        rr = sess.run(self.out, feed_dict={self.inputs: inputs})[0]
        return int(rr[0] > rr[1]), rr


    def train(self, sess, batch_x, batch_y):
        return sess.run([self.optimizer, self.cost], feed_dict={
                self.inputs: batch_x,
                self.labels: batch_y
               })


    def validate(self, num_stages):
        assert(self.layer_activation[-1] == 'softmax') # classification supports only softmax
        assert(self.layer_size[-1] == 2 and num_stages == 2) # Supports only 2-stage curriculum => 2 softmax outputs

###############################################################################
class CurriculumNetwork(object):
    """
    Input to the network is the performance characteristics, output is the prediction switcher.
    """
    def __init__(self, input_dim, config, cl_mode_init):
        network_type = config["cl_structure"].split(":")[0]
        if network_type == 'cl':
            self.network = FeedForwardCurriculumNetwork(input_dim, config)
        if network_type == 'ffsc':
            self.network = FeedForwardSupervisedClassificationNetwork(input_dim, config)

        cl_stages = config["cl_stages"]
        self.stages = cl_stages.split(":")[0].split(";")
        self.constraints = cl_stages.split(":")[1]

        # previous stage
        if cl_mode_init:
            self.stage = self.stages.index(cl_mode_init)
        else:
            self.stage = 0

        self.network.validate(len(self.stages))


    def train(self, sess, batch_x, batch_y):
        if self.network:
            self.network.train(sess, batch_x, batch_y)


    def predict(self, sess, inputs):
        if self.network:
            stage, rr = self.network.predict(sess, inputs, len(self.stages))

            stage = min([len(self.stages)-1, stage])

            if self.constraints == 'monotonic':
                stage = max([self.stage, stage]) # do not allow to go backwards in the curriculum

            self.stage = stage
            return self.stages[stage], rr


    def load(self, sess, fname):
        if self.network:
            return self.network.load(sess, fname)


    def save(self, sess, fname, global_step = None):
        if self.network:
            self.network.save(sess, fname, global_step)
