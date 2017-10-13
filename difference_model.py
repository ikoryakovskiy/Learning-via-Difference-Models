#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import tflearn
import math


class DifferenceModel(object):
    def __init__(self, input_dim, output_dim):

        # NN characteristics
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = 0.02
        self.l2 = 0
        # Defining network
        self.actual_output = tf.placeholder(tf.float32, [None, self.output_dim])
        self.keep_prob = tf.placeholder(tf.float32)
        self.input, self.output, self.layer1, self.layer2, self.layer3 = self.create_network()
        self.network_weights = tf.trainable_variables()

        # Optimizing procedure with L2 regularization
        var = tf.add_n([ tf.nn.l2_loss(v) for v in self.network_weights if 'bias' not in v.name ]) * self.l2
        self.loss = tflearn.mean_square(self.actual_output, self.output) + var
        self.optimize = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def create_network(self):
        input = tflearn.input_data(shape=[None, self.input_dim])
        layer1 = tflearn.fully_connected(input, 400, activation='relu', name="Layer1",weights_init=tflearn.initializations.uniform(
                                             minval=-1 / math.sqrt(self.input_dim),
                                             maxval=1 / math.sqrt(self.input_dim)))
        layer1 = tf.nn.dropout(layer1, self.keep_prob)
        layer2 = tflearn.fully_connected(layer1, 400, activation='relu', name="Layer2",weights_init=tflearn.initializations.uniform(
                                             minval=-1 / math.sqrt(400),
                                             maxval=1 / math.sqrt(400)))
        layer2 = tf.nn.dropout(layer2, self.keep_prob)
        layer3 = tflearn.fully_connected(layer2, 400, activation='relu', name="Layer3",weights_init=tflearn.initializations.uniform(
                                             minval=-1 / math.sqrt(400),
                                             maxval=1 / math.sqrt(400)))
        layer3 = tf.nn.dropout(layer3, self.keep_prob)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        output = tflearn.fully_connected(layer3, self.output_dim, activation='linear', name="Output")
        return input, output, layer1, layer2, layer3

    def train(self, sess, inputs, actual_output, keep_prob=0.5):
        sess.run([self.output, self.optimize], feed_dict={
            self.input: inputs,
            self.actual_output: actual_output,
            self.keep_prob: keep_prob
        })

    def predict(self, sess, inputs, keep_prob=1):
        return sess.run(self.output, feed_dict={
            self.input: inputs,
            self.keep_prob: keep_prob
        })

