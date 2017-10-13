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
        self.learning_rate = 0.01
        self.l2 = 0
        # Defining network
        self.actual_output = tf.placeholder(tf.float32, [None, self.output_dim])
        self.keep_prob = tf.placeholder(tf.float32)
        self.input, self.output, self.net, self.is_training = self.create_network()
        self.network_weights = tf.trainable_variables()

        # Optimizing procedure with L2 regularization
        var = tf.add_n([ tf.nn.l2_loss(v) for v in self.network_weights if 'bias' not in v.name ]) * self.l2
        self.loss = tflearn.mean_square(self.actual_output, self.output) + var
        self.optimize = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def create_network(self):
        layer1_size = 200
        layer2_size = 200
        layer3_size = 200

        input = tf.placeholder("float", [None, self.input_dim])
        is_training = tf.placeholder(tf.bool)

        W1 = self.variable([self.input_dim, layer1_size], self.input_dim)
        b1 = self.variable([layer1_size], self.input_dim)
        W2 = self.variable([layer1_size, layer2_size], layer1_size)
        b2 = self.variable([layer2_size], layer1_size)
        W3 = self.variable([layer2_size, layer3_size], layer2_size)
        b3 = self.variable([layer3_size], layer2_size)
        W4 = self.variable([layer3_size, self.output_dim], layer3_size)
        b4 = self.variable([self.output_dim], layer3_size)

        layer0_bn = self.batch_norm_layer(input, training_phase=is_training, scope_bn='batch_norm_0',
                                          activation=tf.identity)
        layer1 = tf.matmul(layer0_bn, W1) + b1
        layer1_bn = self.batch_norm_layer(layer1, training_phase=is_training, scope_bn='batch_norm_1',
                                          activation=tf.nn.relu)
        layer1_dr = tf.nn.dropout(layer1_bn, self.keep_prob)

        layer2 = tf.matmul(layer1_dr, W2) + b2
        layer2_bn = self.batch_norm_layer(layer2, training_phase=is_training, scope_bn='batch_norm_2',
                                          activation=tf.nn.relu)
        layer2_dr = tf.nn.dropout(layer2_bn, self.keep_prob)

        layer3 = tf.matmul(layer2_dr, W3) + b3
        layer3_bn = self.batch_norm_layer(layer3, training_phase=is_training, scope_bn='batch_norm_3',
                                          activation=tf.nn.relu)
        layer3_dr = tf.nn.dropout(layer3_bn, self.keep_prob)
        output = tf.identity(tf.matmul(layer3_dr, W4) + b4)

        return input, output, [W1, b1, W2, b2, W3, b3, W4, b4], is_training
        # input = tflearn.input_data(shape=[None, self.input_dim])
        # layer1 = tflearn.fully_connected(input, 400, activation='relu', name="Layer1",
        #                                  weights_init=tflearn.initializations.uniform(
        #                                      minval=-1 / math.sqrt(self.input_dim),
        #                                      maxval=1 / math.sqrt(self.input_dim)))
        # layer1 = tf.nn.dropout(layer1, self.keep_prob)
        # layer2 = tflearn.fully_connected(layer1, 400, activation='relu', name="Layer2",
        # layer2 = tf.nn.dropout(layer2, self.keep_prob)
        # layer3 = tflearn.fully_connected(layer2, 400, activation='relu', name="Layer2",
        #                                  weights_init=tflearn.initializations.uniform(minval=-1 / math.sqrt(300),
        #                                                                               maxval=1 / math.sqrt(300)))
        # layer3 = tf.nn.dropout(layer3, self.keep_prob)
        # # Final layer weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # output = tflearn.fully_connected(layer3, self.output_dim, activation='linear', weights_init=w_init,
        #                                  name="Output")
        # return input, output, layer1, layer2, layer3

    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))

    def batch_norm_layer(self, x, training_phase, scope_bn, activation=None):
        return tf.cond(training_phase,
                       lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                                                            updates_collections=None, is_training=True, reuse=None,
                                                            scope=scope_bn, decay=0.99, epsilon=1e-5),
                       lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                                                            updates_collections=None, is_training=False, reuse=True,
                                                            scope=scope_bn, decay=0.99, epsilon=1e-5))

    def train(self, sess, inputs, actual_output, keep_prob=0.7):
        sess.run([self.output, self.optimize], feed_dict={
            self.input: inputs,
            self.actual_output: actual_output,
            self.keep_prob: keep_prob,
            self.is_training: True
        })

    def predict(self, sess, inputs, keep_prob=1):
        return sess.run(self.output, feed_dict={
            self.input: inputs,
            self.keep_prob: keep_prob,
            self.is_training: False
        })

