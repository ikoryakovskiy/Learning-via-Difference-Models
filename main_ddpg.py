#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:49:02 2017

@author: divyam
"""

import tensorflow as tf
import numpy as np
import tflearn
import yaml
import zmq
import time
import struct
import random
import math
import os.path
import sys
from subprocess import Popen
import signal
import global_params
import multiprocessing
import pdb

from replaybuffer_ddpg import ReplayBuffer
from ExplorationNoise import ExplorationNoise
from actor import ActorNetwork
from critic import CriticNetwork
from difference_model import DifferenceModel

# Path to grl executable
GRL_PATH = '../grl/qt-build/grld'

# ==========================
#   Training Parameters
# ==========================
# Max training steps
# MAX_EPISODES = 50000
# Max episode length
MAX_STEPS_EPISODE = 1010
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor 
GAMMA = 0.99
# Soft target update param
TAU = 0.001
FACTOR = 0
# ===========================
#   Utility Parameters
# ===========================

# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
TRAINING_SIZE = 2000
BUFFER_SIZE = 300000
MINIBATCH_SIZE = 64
MIN_BUFFER_SIZE = 20000
# Environment Parameters
ACTION_DIMS = 6
ACTION_DIMS_REAL = 9
STATE_DIMS = 18
OBSERVATION_DIMS = 14
ACTION_BOUND = 1
ACTION_BOUND_REAL = 8.6
# Noise Parameters
NOISE_MEAN = 0
NOISE_VAR = 1
# Ornstein-Uhlenbeck variables
OU_THETA = 0.15
OU_MU = 0
OU_SIGMA = 0.2


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


# ===========================
# Policy saving and loading
# ===========================


def open_config_file(conf):
    with open(conf, 'r') as f:
        config = yaml.load(f)
    return config


def check_for_policy_load(sess, config):
    if "load_file" in config["experiment"]:
        load_file = config["experiment"]["load_file"]
        path = os.path.dirname(os.path.abspath(__file__))
        load_file = "{}/{}".format(path,load_file)
        meta_file = "{}.meta".format(load_file)
        print meta_file
        #pdb.set_trace()
        if os.path.isfile(meta_file):
            saver = tf.train.Saver()
            saver.restore(sess, load_file)
            print "Model Restored"
        else:
            print "Not a valid path"
            sess.run(tf.global_variables_initializer())
    else:
        sess.run(tf.global_variables_initializer())

    return sess


def check_for_policy_save(config):
    save_every = config["experiment"]["save_every"]
    randomize = config["experiment"]["environment"]["task"]["randomize"]

    if save_every == "never":
        save_counter = 0
    elif save_every == "trail":
        save_counter = 1
    elif save_every == "test":
        save_counter = config["experiment"]["test_interval"] + 1
    else:
        save_counter = 10
    return save_counter, randomize


def compute_action(sess, test_agent, randomize, actor, mod_state, noise):
    if test_agent and not randomize:
        action = actor.predict(sess, np.reshape(mod_state, (1, actor.s_dim)))
        # time.sleep(0.05)
    else:
        action = actor.predict(sess, np.reshape(mod_state, (1, actor.s_dim))) + noise

    action = np.reshape(action, (ACTION_DIMS,))

    action = np.clip(action, -1, 1)

    return action


def compute_diff_state_dropout(diff_sess, model, input):
    probs = []
    l = 10
    N = 8000
    p = 0.01
    decay = 0.001
    for _ in xrange(1):
        probs += [model.predict(diff_sess, input)]
    predictive_mean = np.reshape(model.predict(diff_sess, input, 1), (STATE_DIMS,))

    predictive_variance = np.reshape(np.var(probs, axis=0), (STATE_DIMS,))
    tau = l ** 2 * (1 - p) / (2 * N * decay)
    predictive_variance += tau ** -1
    # print predictive_variance
    return predictive_mean, predictive_variance


def get_address(config):
    address = config['experiment']['agent']['communicator']['addr']
    address = address.split(':')[-1]
    address = "tcp://*:{}".format(address)
    return address


def invert(state):
    obs = np.zeros(OBSERVATION_DIMS)
    count = 0
    for i in range(2, STATE_DIMS / 2):
        obs[count] = state[i]
        obs[count + OBSERVATION_DIMS / 2] = state[i + STATE_DIMS / 2]
        count += 1
    return obs


def calculate_new_reward(state, reward):
    reward += 300 * state[0]
    return reward


def calculate_new_terminal(state):

    torsoConstraint = 1
    stanceConstraint = 0.36 * math.pi
    torsoHeightConstraint = -0.15

    if ((abs(state[2]) > torsoConstraint) or (abs(state[8]) > stanceConstraint) or (abs(state[7]) > stanceConstraint)
            or (state[1] < torsoHeightConstraint) or (state[6] > 0) or (state[5] > 0)):
        return 2
    else:
        return 0

# ===========================
#   Agent Training
# ===========================


def train(args, ddpg, actor, critic, counter=None, diff_model=None, model=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    with tf.Session(graph=diff_model,config=tf.ConfigProto(gpu_options=gpu_options)) as diff_sess:
        if model:
            saver = tf.train.Saver()
            saver.restore(diff_sess, "./difference-model")
            saver.save(diff_sess, "difference-model-{}".format(counter))
            print "Difference model restored"
            # saver.save(diff_sess, "difference-model-{}".format(counter))
        with tf.Session(graph=ddpg,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # Start the GRL code
            code = Popen([GRL_PATH, args]) #, '-vv'])

            # Parse the configuration file
            config = open_config_file(args)

            # Check if a policy needs to be loaded
            sess = check_for_policy_load(sess, config)

            # Check if a policy needs to be saved
            save_counter, randomize = check_for_policy_save(config)
            print "Save counter:", save_counter
            print "Noise sigma:", global_params.ou_sigma
            print "Actor learning rate", global_params.actor_learning_rate
            print "Critic learning rate", global_params.critic_learning_rate
            # Initialize target network weights
            actor.update_target_network(sess)
            critic.update_target_network(sess)

            # Initialize replay memory
            if model:
                replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED, diff_sess)
            else:
                replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
            # replay_buffer_test = ReplayBuffer(TRAINING_SIZE, RANDOM_SEED)

            # Initialize constants for exploration noise
            episode_count = 0
            ou_sigma = global_params.ou_sigma
            ou_theta = global_params.ou_theta
            ou_mu = OU_MU
            successful_episodes = 0
            test_reward = 0
            max_test_reward = 0

            # Establish the connection
            context = zmq.Context()
            server = context.socket(zmq.REP)
            server.bind(get_address(config))

            obs_old = np.zeros(actor.s_dim)
            state_old = np.zeros(STATE_DIMS)
            computed_action = np.zeros(ACTION_DIMS)
            check = False

            while True:

                ep_reward = 0
                terminal = 0
                terminal_grl = 0
                terminal_true = 0

                while True:

                    OriginalHandler = signal.signal(signal.SIGALRM, timeout_handler)

                    # Receive the state from zeromq, the current state
                    signal.alarm(3000)
                    try:
                        incoming_message = server.recv()
                    except TimeoutException:
                        print ("No state received from GRL")
                        code.kill()
                        return

                    # Get the length of the message
                    len_incoming_message = len(incoming_message)

                    # Decide which method sent the message and extract the message in a numpy array
                    if len_incoming_message == (STATE_DIMS + 1) * 8:
                        a = np.asarray(struct.unpack('d' * (STATE_DIMS + 1), incoming_message))
                        test_agent = a[0]
                        state = a[1: STATE_DIMS + 1]
                        obs = invert(state)
                        episode_start = True
                        episode_count += 1
                        noise = np.zeros(actor.a_dim)
                    elif len_incoming_message == (STATE_DIMS + 3) * 8:
                        a = np.asarray(struct.unpack('d' * (STATE_DIMS + 3), incoming_message))
                        test_agent = a[0]
                        state = a[1: STATE_DIMS + 1]
                        obs = invert(state)
                        reward = a[STATE_DIMS + 1]
                        terminal_grl = a[STATE_DIMS + 2]
                        episode_start = False
                    else:
                        raise ValueError('DDPG Incomming Zeromq message has a wrong length')

                    # Call to see if the difference model should be used to obtain the true state
                    if model and not episode_start:
                        diff_state, diff_state_variance = compute_diff_state_dropout(diff_sess, model, np.reshape(
                            np.concatenate((np.zeros(1),state_old[1:STATE_DIMS],computed_action)),(1, STATE_DIMS+ACTION_DIMS)))
                    else:
                        diff_state = np.zeros(STATE_DIMS)
                        diff_state_variance = np.zeros(STATE_DIMS)
                    state += diff_state

                    if model and not episode_start:
                        terminal = calculate_new_terminal(state)
                        if terminal and not terminal_grl:
                            reward += -125
                        if terminal_grl and not terminal:
                            reward += 125
                        terminal_grl = terminal

                    obs = invert(state)
                    diff_obs = invert(diff_state)
                    # Add the transition to replay buffer
                    if not episode_start:
                        if test_agent:
                            if terminal_grl == 2:
                                check = replay_buffer.transitions_buffer_add(np.reshape(state_old, (STATE_DIMS,)),
                                                               np.reshape(computed_action, (actor.a_dim,)),
                                                               reward, True, np.reshape(state, (STATE_DIMS,)), diff_state)
                            else:
                                check = replay_buffer.transitions_buffer_add(np.reshape(state_old, (STATE_DIMS,)),
                                                               np.reshape(computed_action, (actor.a_dim,)),
                                                               reward, False, np.reshape(state, (STATE_DIMS,)), diff_state)
                        elif terminal_true == 0:
                            if terminal_grl == 2:
                                check = replay_buffer.replay_buffer_add(np.reshape(obs_old, (actor.s_dim,)),
                                                                        np.reshape(computed_action, (actor.a_dim,)),
                                                                        reward, True, np.reshape(obs, (actor.s_dim,)), diff_obs)
                            else:
                                check = replay_buffer.replay_buffer_add(np.reshape(obs_old, (actor.s_dim,)),
                                                                        np.reshape(computed_action, (actor.a_dim,)),
                                                                        reward, False, np.reshape(obs, (actor.s_dim,)), diff_obs)
                    if check:
                        print "Transitions saved"
                        code.kill()
                        return

                    # Compute OU noise
                    noise = ExplorationNoise.ou_noise(ou_theta, ou_mu, ou_sigma, noise, ACTION_DIMS)

                    # Compute action
                    computed_action = compute_action(sess, test_agent, randomize, actor, obs, noise)

                    state, computed_action = replay_buffer.sample_state_action(state, computed_action, test_agent, episode_start)

                    # Convert state to obs
                    # state = invert(state)

                    # Get state and action from replay buffer to send to GRL
                    scaled_action = computed_action * ACTION_BOUND_REAL

                    if ACTION_DIMS_REAL != ACTION_DIMS:
                        scaled_action = np.concatenate((np.zeros((ACTION_DIMS_REAL - ACTION_DIMS,)), scaled_action))

                    # Convert state and action into null terminated string
                    outgoing_array = np.concatenate((scaled_action, state))
                    outgoing_message = struct.pack('d' * (ACTION_DIMS_REAL + STATE_DIMS), *outgoing_array)

                    # Sends the predicted action via zeromq
                    server.send(outgoing_message)

                    # Keep adding experience to the memory until
                    # there are at least minibatch size samples
                    if not test_agent:
                        if replay_buffer.size() > MIN_BUFFER_SIZE:
                            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                                replay_buffer.sample_batch(MINIBATCH_SIZE)

                            # Calculate targets
                            target_q = critic.predict_target(sess, s2_batch, actor.predict_target(sess, s2_batch))

                            y_i = []
                            for k in xrange(MINIBATCH_SIZE):
                                if t_batch[k]:
                                    y_i.append(r_batch[k])
                                else:
                                    y_i.append(r_batch[k] + GAMMA * target_q[k])

                            # Update the critic given the targets
                            predicted_q_value, _ = critic.train(sess, s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                            # Update the actor policy using the sampled gradient
                            a_outs = actor.predict(sess, s_batch)
                            grads = critic.action_gradients(sess, s_batch, a_outs)
                            actor.train(sess, s_batch, grads[0])

                            # Update target networks
                            actor.update_target_network(sess)
                            critic.update_target_network(sess)

                    # old_state = state

                    obs_old = obs
                    state_old = state
                    terminal_true = terminal_grl


                    if not episode_start:
                        ep_reward += reward
                        if test_agent:
                            test_reward += reward

                    if episode_start and episode_count != 1:
                        print "Episode ended (return, episode_counter, test_return, max_test_return) = ({:>11.3f}, {:>11}, {:>11.3f}, {:>11.3f})".format(ep_reward, episode_count, test_reward, max_test_reward)
                        # if test_reward > 1600:
                        if global_params.test_run_on_model:
                            if test_reward > 1000:
                                global_params.learning_success = 1
                               
                        if save_counter != 0:
                        #        successful_episodes += 1
                            if test_reward > max_test_reward:
                                max_test_reward = test_reward
                                saver = tf.train.Saver()
                                if model:
                                    saver.save(sess, "model-leo-rbdl-with-diff-{}.ckpt".format(counter))
                                else:
                                    pdb.set_trace()
                                    saver.save(sess, "model-leo-rbdl-{}.ckpt".format(1))
                        test_reward = 0
                        break

                if global_params.test_run_on_model:
                    print "Run terminated"
                    code.kill()
                    break


def start(args, counter=None):

    # Initialize the actor, critic and difference networks

    with tf.Graph().as_default() as ddpg:
        actor = ActorNetwork(OBSERVATION_DIMS, ACTION_DIMS, 1,
                             global_params.actor_learning_rate, TAU)
        critic = CriticNetwork(OBSERVATION_DIMS, ACTION_DIMS, global_params.critic_learning_rate, TAU,
                               actor.get_num_trainable_vars())
    if counter:
        with tf.Graph().as_default() as diff_model:
            model = DifferenceModel(STATE_DIMS+ACTION_DIMS, STATE_DIMS)
            train(args, ddpg, actor, critic, counter=counter, diff_model=diff_model, model=model)
    else:
        train(args, ddpg, actor, critic)



