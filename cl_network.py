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
import collections
import pickle

###############################################################################
class NeuralNetwork(object):
    target_network_params = []
    network_params = []

    def __init__(self, input_dim, action_dim, config):
        self.i_dim = input_dim
        self.a_dim = action_dim
        self.learning_rate = config["cl_lr"]
        self.l2 = config["cl_l2_reg"]
        self.tau = config["cl_tau"]
        self.dropout = config["cl_dropout_keep"]
        self.batch_norm = config["cl_batch_norm"]
        self.structure = config["cl_structure"]
        self.network_type = config["cl_structure"].split(":")[0]
        self._decode()
        if 'critic' in self.network_type:
            self.inputs, self.action, self.out = self._create_critic('curriculum')
        else:
            self.inputs, self.out = self._create('curriculum')
        self.network_params = [v for v in tf.trainable_variables() if 'curriculum' in v.name]

        if 'critic' in self.network_type and config["cl_target"]:
            self.target_inputs, self.target_action, self.target_out = self._create_critic(prefix='cltg')
            self.target_network_params = [v for v in tf.trainable_variables() if 'cltg' in v.name]
            self.update_target_network_params = \
                [self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i],
                                                                                1. - self.tau))
                 for i in range(len(self.target_network_params))]

    def load(self, sess, fname):
        if exists(fname+'.npy'):
            params = np.load(fname+'.npy').squeeze()
            self._set_params(sess, params)
        else:
            toload = self.target_network_params + self.network_params
            saver = tf.train.Saver(toload)
            saver.restore(sess, fname)
        print("Loaded curriculum from {}".format(fname))
        return sess



    def save(self, sess, fname, global_step = None):
        tosave = self.target_network_params + self.network_params
        saver = tf.train.Saver(tosave, max_to_keep=None)
        saver.save(sess, "./" + fname, global_step)


    def _decode(self):
        self.layer_size = [self.i_dim[-1]] # includes input layer
        self.layer_type = ['']
        self.layer_activation = ['']
        self.layer_other = ['']
        self.w_num = 0 # total number of weights in NN
        network_description = self.structure.split(":")[1].split(";")
        for layer in network_description:
            prop = layer.split("_")
            ltype, activation, size = prop[0], prop[1], prop[2]
            self.w_num += self.layer_size[-1]*int(size) + int(size)
            self.layer_type.append(ltype)
            self.layer_activation.append(activation)
            self.layer_size.append(int(size))
            self.layer_other.append(prop[3:] if len(prop)>3 else '')
        self.num_hidden_layers = len(self.layer_size)-1


    def _create(self, prefix=''):
        shape = [None] + [i for i in self.i_dim]
        inputs = tflearn.input_data(shape=shape)
        layer = inputs

#        init_state = tf.get_variable('{}Initstate'.format(prefix), [1, 6],
#                                 initializer=tf.constant_initializer(0.0))
#        init_state = tf.tile(init_state, [2, 1])

        for i in range(self.num_hidden_layers):
            weights_init = tflearn.initializations.uniform(minval=-1/sqrt(self.layer_size[i]), maxval=1/sqrt(self.layer_size[i]))

            if 'dropout' in self.layer_other[i+1]:
                dropout = self.dropout
            else:
                dropout = None

            if self.layer_type[i+1] == 'fc':
                new_layer = tflearn.fully_connected(layer, self.layer_size[i+1], name="{}Layer{}".format(prefix,i), weights_init=weights_init)
            elif self.layer_type[i+1] == 'rnn':
                new_layer = tflearn.simple_rnn(layer, self.layer_size[i+1], name="{}Layer{}".format(prefix,i),
                                               weights_init=weights_init,
                                               return_seq=False,
                                               activation='linear',
                                               dropout=dropout,
                                               #initial_state=init_state,
                                               dynamic=True)
            elif self.layer_type[i+1] == 'gru':
                new_layer = tflearn.gru(layer, self.layer_size[i+1], name="{}Layer{}".format(prefix,i),
                                               weights_init=weights_init,
                                               return_seq=False,
                                               activation='linear',
                                               dropout=dropout,
                                               #initial_state=init_state,
                                               dynamic=True)
            elif self.layer_type[i+1] == 'lstm':
                new_layer = tflearn.lstm(layer, self.layer_size[i+1], name="{}Layer{}".format(prefix,i),
                                               weights_init=weights_init,
                                               return_seq=False,
                                               activation='linear',
                                               dropout=dropout,
                                               dynamic=True)
            else:
                raise ValueError('Unsupported layer {}'.format(i))

            if self.batch_norm:
                new_layer = tflearn.layers.normalization.batch_normalization(new_layer, name="{}Layer{}_norm".format(prefix,i))

            if self.layer_activation[i+1] == 'linear':
                new_layer = tflearn.activations.linear(new_layer)
            elif self.layer_activation[i+1] == 'relu':
                new_layer = tflearn.activations.relu(new_layer)
            elif self.layer_activation[i+1] == 'tanh':
                new_layer = tflearn.activations.tanh(new_layer)
            elif self.layer_activation[i+1] == 'sigmoid':
                new_layer = tflearn.activations.sigmoid(new_layer)

            if i < self.num_hidden_layers-1:
                layer = new_layer
        return inputs, new_layer


    def _create_critic(self, prefix=''):
        inputs_shape = [None] + [i for i in self.i_dim]
        inputs = tflearn.input_data(shape=inputs_shape)

        action_shape = [None] + [i for i in self.a_dim]
        action = tflearn.input_data(shape=action_shape)

        layer = inputs
        for i in range(self.num_hidden_layers):
            weights_init = tflearn.initializations.uniform(minval=-1/sqrt(self.layer_size[i]), maxval=1/sqrt(self.layer_size[i]))

            if 'dropout' in self.layer_other[i+1]:
                dropout = self.dropout
            else:
                dropout = None

            if self.layer_type[i+1] == 'fc':
                new_layer = tflearn.fully_connected(layer, self.layer_size[i+1], name="{}Layer{}".format(prefix,i), weights_init=weights_init)
            elif self.layer_type[i+1] == 'rnn':
                new_layer = tflearn.simple_rnn(layer, self.layer_size[i+1], name="{}Layer{}".format(prefix,i),
                                               weights_init=weights_init,
                                               return_seq=False,
                                               activation='linear',
                                               dropout=dropout,
                                               dynamic=True)
            else:
                raise ValueError('Unsupported layer {}'.format(i))

            if i == self.num_hidden_layers-2: # last layer is actor
                 break

            if self.batch_norm:
                new_layer = tflearn.layers.normalization.batch_normalization(new_layer, name="{}Layer{}_norm".format(prefix,i))

            if self.layer_activation[i+1] == 'linear':
                new_layer = tflearn.activations.linear(new_layer)
            elif self.layer_activation[i+1] == 'relu':
                new_layer = tflearn.activations.relu(new_layer)
            elif self.layer_activation[i+1] == 'tanh':
                new_layer = tflearn.activations.tanh(new_layer)
            elif self.layer_activation[i+1] == 'sigmoid':
                new_layer = tflearn.activations.sigmoid(new_layer)

            if i < self.num_hidden_layers-1:
                layer = new_layer

        action_init = tflearn.initializations.uniform(minval=-1/sqrt(self.layer_size[-3]),
                                                                     maxval=1/sqrt(self.layer_size[-3]))
        if self.layer_type[-1] == 'fc':
            action_layer = tflearn.fully_connected(action, self.layer_size[-1], name="{}LayerAction".format(prefix), weights_init=action_init)
        else:
            raise ValueError('Unsupported actor layer')

        if self.layer_activation[-1] == 'relu':
            net = tflearn.activation(tf.matmul(layer, new_layer.W) + tf.matmul(action, action_layer.W) +
                                         action_layer.b, activation='relu')
        else:
            raise ValueError('Unsupported actor activation')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        new_layer = tflearn.fully_connected(net, 1, weights_init=w_init, name="{}Output".format(prefix))

        return inputs, action, new_layer

    def predict_target_(self, sess, batch_x, **kwargs):
        pass


    def update_target_network(self, sess):
        pass

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
class FeedForwardCurriculumNetwork(NeuralNetwork):
    def __init__(self, input_dim, config, num_stages):
        super().__init__(input_dim, [0], config)
        self.num_stages = num_stages

    def predict(self, sess, inputs):
        rr = self.predict_(sess, inputs, self.num_stages)
        if self.layer_activation[-1] == 'tanh':
            eps = 1E-7
            bins = np.linspace(-1-eps, 1+eps, self.num_stages+1) # requires tanh output layer of NN
            idx = np.digitize(rr, bins, right=True)[0] - 1 # index starts at 1, and include 0 as a left bin
        else:
            reversed_rr = rr[::-1]
            binary = map(int, reversed_rr > 0)
            binary_string = ''.join(map(str, binary))
            idx = int(binary_string, 2)
        return idx, rr

    def predict_(self, sess, inputs, **kwargs):
        rr = sess.run(self.out, feed_dict={self.inputs: inputs})[0]
        return rr

    def train(self, sess, batch_x, batch_y, **kwargs):
        # curriculum network is trained by metaheuristic
        pass

    def validate(self):
        assert((self.layer_size[-1] == 1 and self.num_stages == 2) or # 2-stage curriculum
               (self.layer_size[-1] == 2 and self.num_stages == 3))   # 3-stage curriculum


###############################################################################
class FeedForwardSupervisedClassificationNetwork(NeuralNetwork):
    def __init__(self, input_dim, config, num_stages):
        super().__init__(input_dim, [0], config)
        self.num_stages = num_stages
        self.l2 = config["cl_l2_reg"]
        self.learning_rate = config["cl_lr"]

        self.labels = tf.placeholder("float", [None, self.layer_size[-1]])
        var = tf.add_n([ tf.nn.l2_loss(v) for v in self.network_params if 'bias' not in v.name ]) * self.l2
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.labels) + var)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

    def predict(self, sess, inputs):
        rr = self.predict_(sess, inputs, self.num_stages)
        return int(rr[0] > rr[1]), rr

    def predict_(self, sess, inputs, **kwargs):
        rr = sess.run(self.out, feed_dict={self.inputs: inputs})[0]
        return rr

    def train(self, sess, batch_x, batch_y, **kwargs):
        return sess.run([self.optimizer, self.cost], feed_dict={
                self.inputs: batch_x,
                self.labels: batch_y
               })

    def validate(self):
        assert(self.layer_activation[-1] == 'softmax') # classification supports only softmax
        assert(self.layer_size[-1] == 2 and self.num_stages == 2) # Supports only 2-stage curriculum => 2 softmax outputs


###############################################################################
class RecurrentNeuralClassificationNetwork(NeuralNetwork):
    def __init__(self, input_dim, config, num_stages):
        super().__init__(input_dim, [0], config)
        self.num_stages = num_stages
        self.l2 = config["cl_l2_reg"]
        self.learning_rate = config["cl_lr"]

        self.labels = tf.placeholder("float", [None, self.layer_size[-1]])
        self.class_weight = tf.placeholder("float", [None, 1])
        #not_bias = [v for v in self.network_params if ('/Bias:' not in v.name and '/b:' not in v.name)]
        #var = tf.add_n([tf.nn.l2_loss(v) for v in not_bias]) * self.l2

        # Take the cost like normal
        error = tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.labels)

        # Scale the cost by the class weights
        scaled_error = tf.multiply(error, self.class_weight)

        self.cost = tf.reduce_mean(scaled_error)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)


    def predict(self, sess, inputs):
        rr = self.predict_(sess, inputs)
        label = np.argmax(rr)
        return label, rr

    def predict_(self, sess, inputs, **kwargs):
        rr = sess.run(self.out, feed_dict={self.inputs: inputs})[0]
        return rr

    def train(self, sess, batch_x, batch_y, **kwargs):
        class_weight = kwargs['class_weight']
        return sess.run([self.optimizer, self.cost], feed_dict={
                self.inputs: batch_x,
                self.labels: batch_y,
                self.class_weight: class_weight
               })

    def validate(self):
        assert(self.layer_activation[-1] == 'linear') # classification supports only logits
        #assert(self.layer_size[-1] == 2 and self.num_stages == 2) # Supports only 2-stage curriculum => 2 softmax outputs


###############################################################################
class FeedForwardRegressionNetwork(NeuralNetwork):
    def __init__(self, input_dim, config, num_stages):
        super().__init__(input_dim, [0], config)
        self.num_stages = num_stages
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.l2_reg = tf.add_n([ tf.nn.l2_loss(v) for v in self.network_params if '/b:' not in v.name ]) * self.l2

        self.loss = tflearn.mean_square(self.predicted_q_value, self.out) + self.l2_reg
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def predict(self, sess, inputs):
        rr = []
        for action in range(self.num_stages):
            action = np.reshape(action-1, [-1, 1])
            inputs_new = np.concatenate((inputs, action), axis=1)
            rr.append(self.predict_(sess, inputs_new))

        # take curriculum which leads to the higherst return (least damage)
        idx = np.argmax(rr)
        rr = np.reshape(rr, [-1])
        return idx, rr

    def predict_(self, sess, batch_x, **kwargs):
        outputs = sess.run(self.out, feed_dict={
            self.inputs: batch_x
        })
        return outputs

    def train(self, sess, batch_x, batch_y, **kwargs):
        return sess.run([self.out, self.optimize], feed_dict={
            self.inputs: batch_x,
            self.predicted_q_value: batch_y
        })

    def validate(self):
        pass

###############################################################################
class RecurrentNeuralRegressionNetwork(NeuralNetwork):
    def __init__(self, input_dim, config, num_stages):
        super().__init__(input_dim, [0], config)
        self.num_stages = num_stages
        self.optimizer = tflearn.regression(self.out, optimizer='adam', loss='mean_square', learning_rate=self.learning_rate)

    def predict(self, sess, inputs):
        rr = []
        for i in range(self.num_stages):
            inputs[:, :, -1] = (i+1)/10
            rr.append(self.predict_(sess, inputs, self.num_stages))

        # take curriculum which leads to the least damage
        idx = np.argmin(rr)
        return idx, rr

    def predict_(self, sess, inputs, **kwargs):
        outputs = self.model.predict(inputs)
        return outputs

    def train(self, sess, batch_x, batch_y, **kwargs):
        self.model = tflearn.DNN(self.optimizer, tensorboard_verbose=0) # clip_gradients=0.0 disables clipping
        self.model.fit(batch_x, batch_y, **kwargs)

    def validate(self):
        pass

###############################################################################
class FeedForwardCriticNetwork(NeuralNetwork):
    def __init__(self, input_dim, action_dim, config, num_stages):
        super().__init__(input_dim, action_dim, config)
        self.num_stages = num_stages
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.l2_reg = tf.add_n([ tf.nn.l2_loss(v) for v in self.network_params if '/b:' not in v.name ]) * self.l2

        self.loss = tflearn.mean_square(self.predicted_q_value, self.out) + self.l2_reg
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def predict(self, sess, inputs):
        rr = []
        for action in range(self.num_stages):
            action = np.reshape(action-1, [-1, 1])
            rr.append(self.predict_(sess, inputs, action=action))

        # take curriculum which leads to the higherst return (least damage)
        idx = np.argmax(rr)
        rr = np.reshape(rr, [-1])
        return idx, rr

    def predict_(self, sess, batch_x, **kwargs):
        action = kwargs['action']
        outputs = sess.run(self.out, feed_dict={
            self.inputs: batch_x,
            self.action: action
        })
        return outputs


    def predict_target_(self, sess, batch_x, **kwargs):
        action = kwargs['action']
        return sess.run(self.target_out, feed_dict={
            self.target_inputs: batch_x,
            self.target_action: action
        })


    def update_target_network(self, sess):
        sess.run(self.update_target_network_params)


    def train(self, sess, batch_x, batch_y, **kwargs):
        action = kwargs['action']
        return sess.run([self.out, self.optimize], feed_dict={
            self.inputs: batch_x,
            self.action: action,
            self.predicted_q_value: batch_y
        })

    def validate(self):
        pass

###############################################################################
class CurriculumNetwork(object):
    """
    Input to the network is the performance characteristics, output is the prediction switcher.
    """
    def __init__(self, input_dim, config, cl_mode_init = None):
        if not isinstance(input_dim, collections.Sequence):
            input_dim = [input_dim]
        network_type = config["cl_structure"].split(":")[0]
        cl_stages = config["cl_stages"]
        self.stages = cl_stages.split(":")[0].split(";")
        self.constraints = cl_stages.split(":")[1]
        num_stages = len(self.stages)

        if network_type == 'cl':
            self.network = FeedForwardCurriculumNetwork(input_dim, config, num_stages)
        elif network_type == 'ffsc':
            self.network = FeedForwardSupervisedClassificationNetwork(input_dim, config, num_stages)
        elif network_type == 'rnnc':
            self.network = RecurrentNeuralClassificationNetwork(input_dim, config, num_stages)
        elif network_type == 'ffr':
            self.network = FeedForwardRegressionNetwork(input_dim, config, num_stages)
        elif network_type == 'rnnr':
            self.network = RecurrentNeuralRegressionNetwork(input_dim, config, num_stages)
        elif network_type == 'rnncritic':
            pass
            #self.network = RecurrentNeuralCriticNetwork(input_dim, config, num_stages)
        elif network_type == 'ffcritic':
            self.network = FeedForwardCriticNetwork(input_dim, [1], config, num_stages)

        # previous stage
        if cl_mode_init:
            self.stage = self.stages.index(cl_mode_init)
        else:
            self.stage = 0

        self.network.validate()


    def train(self, sess, batch_x, batch_y, **kwargs):
        if self.network:
            return self.network.train(sess, batch_x, batch_y, **kwargs)


    def predict(self, sess, inputs):
        if self.network:
            stage, rr = self.network.predict(sess, inputs)

            stage = min([len(self.stages)-1, stage])

            if self.constraints == 'monotonic':
                stage = max([self.stage, stage]) # do not allow to go backwards in the curriculum

            self.stage = stage
            return self.stages[stage], rr


    def predict_(self, sess, inputs, **kwargs):
        if self.network:
            return self.network.predict_(sess, inputs, **kwargs)

    def predict_target_(self, sess, batch_x, **kwargs):
        if self.network:
            return self.network.predict_target_(sess, batch_x, **kwargs)

    def update_target_network(self, sess):
        self.network.update_target_network(sess)


    def load(self, sess, fname):
        if self.network:
            return self.network.load(sess, fname)


    def save(self, sess, fname, global_step = None):
        if self.network:
            self.network.save(sess, fname, global_step)
