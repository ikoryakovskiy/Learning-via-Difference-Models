#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:49:02 2017

@author: divyam
"""

import tensorflow as tf
import tflearn
from math import sqrt
import pdb

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, state_dim, action_dim, config, num_actor_vars):
        # self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = config["critic_lr"]
        self.tau = config["tau"]
        self.l2 = config["critic_l2_reg"]
        self.batch_norm = config["batch_norm"]
        self.version = config["version"]

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]
        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network(prefix='target_')

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i],
                                                                            1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        var = tf.add_n([ tf.nn.l2_loss(v) for v in self.network_params if 'bias' not in v.name ]) * self.l2

        self.loss = tflearn.mean_square(self.predicted_q_value, self.out) + var
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self, prefix=''):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        weights_init1 = tflearn.initializations.uniform(minval=-1/sqrt(self.s_dim), maxval=1/sqrt(self.s_dim))
        critic_layer1 = tflearn.fully_connected(inputs, 400, name="{}criticLayer1".format(prefix), weights_init=weights_init1)
        if self.batch_norm:
            critic_layer1 = tflearn.layers.normalization.batch_normalization(critic_layer1, name="{}criticLayer1_norm".format(prefix))
        critic_layer1_relu = tflearn.activations.relu(critic_layer1)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        weights_init2 = tflearn.initializations.uniform(minval=-1/sqrt(400 + self.a_dim), maxval=1/sqrt(400 + self.a_dim))
        critic_layer2 = tflearn.fully_connected(critic_layer1_relu, 300, name="{}criticLayer2".format(prefix),weights_init=weights_init2)

        weights_init3 = tflearn.initializations.uniform(minval=-1/sqrt(400 + self.a_dim), maxval=1/sqrt(400 + self.a_dim))
        critic_layer3 = tflearn.fully_connected(action, 300, name="{}criticLayerAction".format(prefix), weights_init=weights_init3)

        #pdb.set_trace()
        if self.version == 0 or not self.batch_norm:
            net = tflearn.activation(tf.matmul(critic_layer1_relu, critic_layer2.W) + tf.matmul(action, critic_layer3.W) +
                                     critic_layer3.b, activation='relu')
        elif self.version > 0:
            # ivan 6.01.2018 adding another normailization layer
            net = tf.matmul(critic_layer1_relu, critic_layer2.W) + tf.matmul(action, critic_layer3.W) + critic_layer3.b
            net = tflearn.layers.normalization.batch_normalization(net, name="{}criticLayer2_norm".format(prefix))
            net = tflearn.activations.relu(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        critic_output = tflearn.fully_connected(net, 1, weights_init=w_init, name="{}criticOutput".format(prefix))
        return inputs, action, critic_output

    def train(self, sess, inputs, action, predicted_q_value):
        return sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, sess, inputs, action):
        return sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, sess, inputs, action):
        return sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, sess, inputs, actions):
        return sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self, sess):
        sess.run(self.update_target_network_params)

