#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:49:02 2017

@author: divyam, ikoryakovskiy
"""

import tensorflow as tf
import numpy as np
import yaml
import zmq
import struct
import os.path
import subprocess
import signal
import time
from datetime import datetime
import sys

from replaybuffer_ddpg import ReplayBuffer
from ExplorationNoise import ExplorationNoise
from actor import ActorNetwork
from critic import CriticNetwork
from difference_model import DifferenceModel

# ==========================
#   Environment Parameters
# ==========================
# Environment Parameters
ACTION_DIMS = 6
ACTION_DIMS_REAL = 9
STATE_DIMS = 18
OBSERVATION_DIMS = 14
ACTION_BOUND = 1
ACTION_BOUND_REAL = 8.6 # in voltage

# GRL as a global variable
GRL_PATH = '../grl/qt-build/grld'
grl = None

# ===========================
# ZeroMQ timeout handler
# ===========================
class TimeoutException(Exception):
    pass


def timeout_handler():
    raise TimeoutException

def signal_handler(signal, frame):
    if grl.poll() is None:
        grl.kill()
    sys.exit(0)

# ===========================
# Processing YAML configuration
# ===========================
def read_cfg(cfg):
    with open(cfg, 'r') as f:
        conf = yaml.load(f)
        print("Loaded configuration from {}".format(cfg))

        trials = conf["experiment"]["trials"]
        steps = conf["experiment"]["steps"]
        test_interval = conf["experiment"]["test_interval"]
        output = conf["experiment"]["output"]
        save_every = conf["experiment"]["save_every"]
        randomize = conf["experiment"]["environment"]["task"]["randomize"]
        address = conf['experiment']['agent']['communicator']['addr']

        load_file = None
        if "load_file" in conf["experiment"]:
            load_file = conf["experiment"]["load_file"]

        config = dict(trials=trials, steps=steps, test_interval=test_interval, output=output, save_every=save_every,
                        randomize=randomize, load_file=load_file, address=address)

        params = conf["ddpg_param"]
    return config, params

# ===========================
# Policy saving and loading
# ===========================
def preload_policy(sess, config):
    if config["load_file"]:
        load_file = config["load_file"]
        path = os.path.dirname(os.path.abspath(__file__))
        load_file = "{}/{}".format(path, load_file)
        meta_file = "{}.meta".format(load_file)
        print("Loading meta file {}".format(meta_file))
        if os.path.isfile(meta_file):
            saver = tf.train.Saver()
            saver.restore(sess, load_file)
            print("Model Restored")
        else:
            print("Not a valid path")
            sess.run(tf.global_variables_initializer())
    else:
        sess.run(tf.global_variables_initializer())
    return sess


def get_policy_save_counter(config):
    save_counter = 0
    if config["output"]:
        if config["save_every"] == "trail":
            save_counter = 1
        elif config["save_every"] == "test":
            save_counter = config["test_interval"] + 1
        else:
            save_counter = 10
    return save_counter


def compute_action(sess, test_agent, randomize, actor, mod_state, noise):
    if test_agent and not randomize:
        action = actor.predict(sess, np.reshape(mod_state, (1, actor.s_dim)))
    else:
        action = actor.predict(sess, np.reshape(mod_state, (1, actor.s_dim))) + noise
    action = np.reshape(action, (ACTION_DIMS,))
    action = np.clip(action, -1, 1)
    return action


def compute_diff_state_dropout(diff_sess, model, v):
    probs = []
    l = 10
    n = 8000
    p = 0.01
    decay = 0.001
    for _ in range(1):
        probs += [model.predict(diff_sess, v)]
    predictive_mean = np.reshape(model.predict(diff_sess, v, 1), (STATE_DIMS,))

    predictive_variance = np.reshape(np.var(probs, axis=0), (STATE_DIMS,))
    tau = l ** 2 * (1 - p) / (2 * n * decay)
    predictive_variance += tau ** -1
    # print predictive_variance
    return predictive_mean, predictive_variance


def get_address(addr):
    address = addr.split(':')[-1]
    address = "tcp://*:{}".format(address)
    return address


def observe(state):
    obs = np.zeros(OBSERVATION_DIMS)
    count = 0
    for i in range(2, STATE_DIMS / 2):
        obs[count] = state[i]
        obs[count + OBSERVATION_DIMS / 2] = state[i + STATE_DIMS / 2]
        count += 1
    return obs


# ===========================
#   Agent Training
# ===========================
def train(cfg, ddpg, actor, critic, config, params, counter=None, diff_model=None, model=None):
    global grl
    assert(actor.a_dim == ACTION_DIMS)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.signal(signal.SIGINT, signal_handler)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    with tf.Session(graph=diff_model, config=tf.ConfigProto(gpu_options=gpu_options)) as diff_sess:
        if model:
            saver = tf.train.Saver()
            saver.restore(diff_sess, "./difference-model")
            saver.save(diff_sess, "difference-model-{}".format(counter))
            print("Difference model restored")
            # saver.save(diff_sess, "difference-model-{}".format(counter))
        with tf.Session(graph=ddpg, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # Start the GRL code
            grl = subprocess.Popen([GRL_PATH, cfg])

            # Check if a policy needs to be loaded
            sess = preload_policy(sess, config)

            # Check if a policy needs to be saved
            save_counter = get_policy_save_counter(config)

            print("Save counter: {}".format(save_counter))
            print("Noise: {} and {}".format(params["learning"]["ou_sigma"], params["learning"]["ou_theta"]))
            print("Actor learning rate {}".format(params["learning"]["actor_learning_rate"]))
            print("Critic learning rate {}".format(params["learning"]["critic_learning_rate"]))
            print("Minibatch size {}".format(params["replay_buffer"]["minibatch_size"]))

            # Initialize target network weights
            actor.update_target_network(sess)
            critic.update_target_network(sess)

            # Initialize replay memory
            if model:
                replay_buffer = ReplayBuffer(params["replay_buffer"]["max_size"], datetime.now(), diff_sess, params=params)
            else:
                replay_buffer = ReplayBuffer(params["replay_buffer"]["max_size"], datetime.now(), params=params)

            # Initialize constants for exploration noise
            ou_sigma = params["learning"]["ou_sigma"]
            ou_theta = params["learning"]["ou_theta"]
            ou_mu = params["learning"]["ou_mu"]
            trial_return = 0
            max_trial_return = 0

            # Establish the connection
            context = zmq.Context()
            server = context.socket(zmq.REP)
            server.bind(get_address(config["address"]))

            obs = np.zeros(OBSERVATION_DIMS)
            state = np.zeros(STATE_DIMS)
            action = np.zeros(ACTION_DIMS)
            noise = np.zeros(ACTION_DIMS)
            diff_obs = None

            tt = 0
            ss = 0
            terminal = 0

            # Loop over steps and trials, but break is allowed at terminal states only
            while not terminal or \
                    ((not config["trials"] or tt < config["trials"]) and (not config["steps"] or ss < config["steps"])):

                # Receive the current state from zeromq within 3000 seconds
                signal.alarm(3000)
                try:
                    incoming_message = server.recv()
                except TimeoutException:
                    print ("No state received from GRL")
                    grl.kill()
                    return
                finally:
                    signal.alarm(0)

                # Get the length of the message
                len_incoming_message = len(incoming_message)

                # Decide which method sent the message and extract the message in a numpy array
                if len_incoming_message == (STATE_DIMS + 1) * 8:
                    # Message at the beginning of the trial, reward and terminal are missing.
                    a = np.asarray(struct.unpack('d' * (STATE_DIMS + 1), incoming_message))
                    test_agent = a[0]
                    next = a[1: STATE_DIMS + 1]
                    reward = 0
                    terminal = 0
                    trial_start = True
                    # Reset values in the beginning of each new trial
                    trial_return = 0
                    noise = np.zeros(actor.a_dim)
                elif len_incoming_message == (STATE_DIMS + 3) * 8:
                    # Normal message until the end of the trial
                    a = np.asarray(struct.unpack('d' * (STATE_DIMS + 3), incoming_message))
                    test_agent = a[0]
                    next = a[1: STATE_DIMS + 1]
                    reward = a[STATE_DIMS + 1]
                    terminal = a[STATE_DIMS + 2]
                    trial_start = False
                else:
                    raise ValueError('DDPG Incoming zeromq message has a wrong length')

                # for debugging purpose
                test = (config["test_interval"] >= 0 and tt % (config["test_interval"] + 1) == config["test_interval"])
                assert (test_agent == test)

                # Call to see if the difference model should be used to obtain the true state
                if model and not trial_start:
                    diff_next, diff_state_variance = compute_diff_state_dropout(diff_sess, model, np.reshape(
                        np.concatenate((np.zeros(1), state[1:STATE_DIMS], action)),
                        (1, STATE_DIMS + ACTION_DIMS)))
                    next += diff_next
                    diff_obs = observe(diff_next)

                # obtain observation of a state
                next_obs = observe(next)

                # Add the transition to replay buffer
                if not trial_start:
                    replay_buffer.replay_buffer_add(obs, action, reward, terminal == 2, next_obs, diff_obs)

                # Compute OU noise
                noise = ExplorationNoise.ou_noise(ou_theta, ou_mu, ou_sigma, noise, ACTION_DIMS)

                # Compute action
                next_action = compute_action(sess, test_agent, config["randomize"], actor, next_obs, noise)
                next, next_action = replay_buffer.sample_state_action(next, next_action, test_agent, trial_start)

                # Get state and action from replay buffer to send to GRL
                scaled_action = next_action * ACTION_BOUND_REAL

                if ACTION_DIMS_REAL != ACTION_DIMS:
                    scaled_action = np.concatenate((np.zeros((ACTION_DIMS_REAL - ACTION_DIMS,)), scaled_action))

                # Convert state and action into null terminated string
                outgoing_array = np.concatenate((scaled_action, next))
                outgoing_message = struct.pack('d' * (ACTION_DIMS_REAL + STATE_DIMS), *outgoing_array)

                # Sends the predicted action via zeromq
                server.send(outgoing_message)

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if not test_agent and replay_buffer.size() > params["replay_buffer"]["min_size"]:
                    minibatch_size = params["replay_buffer"]["minibatch_size"]
                    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(minibatch_size)

                    # Calculate targets
                    target_q = critic.predict_target(sess, s2_batch, actor.predict_target(sess, s2_batch))

                    y_i = []
                    for k in range(minibatch_size):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + params["learning"]["gamma"] * target_q[k])

                    # Update the critic given the targets
                    #predicted_q_value, _ = \
                    critic.train(sess, s_batch, a_batch, np.reshape(y_i, (minibatch_size, 1)))

                    # Update the actor policy using the sampled gradient
                    a_outs = actor.predict(sess, s_batch)
                    grads = critic.action_gradients(sess, s_batch, a_outs)
                    actor.train(sess, s_batch, grads[0])

                    # Update target networks
                    actor.update_target_network(sess)
                    critic.update_target_network(sess)

                state = next
                obs = next_obs
                action = next_action
                trial_return += reward

                if not test and not trial_start:
                    ss = ss + 1

                if terminal:
                    tt = tt + 1

                    print("Episode ended (return, max_trial_return, tt, ss) = "
                          "({:>11.3f}, {:>11.3f}, {:>11}, {:>11})"
                          .format(trial_return, max_trial_return, tt, ss))

                    if save_counter != 0:
                        if trial_return > max_trial_return:
                            max_trial_return = trial_return
                            saver = tf.train.Saver()
                            if model:
                                saver.save(sess, "./{}-diff-{}".format(config["output"], counter))
                            else:
                                saver.save(sess, "./{}".format(config["output"]))


            # Give GRL some seconds to finish
            time.sleep(1)
            if grl.poll() is None:
                print("GRL will be killed, cfg = {}".format(cfg))
                grl.kill()


def start(cfg, counter=None):
    # Load configuration file
    config, params = read_cfg(cfg)

    # Initialize the actor, critic and difference networks
    with tf.Graph().as_default() as ddpg:
        actor = ActorNetwork(OBSERVATION_DIMS, ACTION_DIMS, 1,
                             params["learning"]["actor_learning_rate"], params["learning"]["tau"])
        critic = CriticNetwork(OBSERVATION_DIMS, ACTION_DIMS,
                               params["learning"]["critic_learning_rate"], params["learning"]["tau"],
                               actor.get_num_trainable_vars())
    if counter:
        with tf.Graph().as_default() as diff_model:
            model = DifferenceModel(STATE_DIMS + ACTION_DIMS, STATE_DIMS)
            train(cfg, ddpg, actor, critic, config, params, counter=counter, diff_model=diff_model, model=model)
    else:
        train(cfg, ddpg, actor, critic, config, params)
