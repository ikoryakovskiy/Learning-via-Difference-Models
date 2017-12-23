#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:49:02 2017

@author: divyam, ikoryakovskiy
"""
import tensorflow as tf
import numpy as np
import os.path
from replaybuffer_ddpg import ReplayBuffer
from ExplorationNoise import ExplorationNoise
from actor import ActorNetwork
from critic import CriticNetwork
import random

# ==========================
#   Environment Parameters
# ==========================
# Environment Parameters
#ACTION_DIMS = 6
#ACTION_DIMS_REAL = 9
#STATE_DIMS = 18
#OBSERVATION_DIMS = 14
#ACTION_BOUND = 1
#ACTION_BOUND_REAL = 8.6

# GRL as a global variable
#GRL_PATH = '../grl/qt-build/grld'

# ===========================
# Policy saving and loading
# ===========================
def preload_policy(sess, saver, config):
    if config["load_file"]:
        load_file = config["load_file"]
        path = os.path.dirname(os.path.abspath(__file__))
        load_file = "{}/{}".format(path, load_file)
        meta_file = "{}.meta".format(load_file)
        print("Loading meta file {}".format(meta_file))
        if os.path.isfile(meta_file):
            saver.restore(sess, load_file)
            print("Model Restored")
        else:
            print("Not a valid path")
            sess.run(tf.global_variables_initializer())
    else:
        sess.run(tf.global_variables_initializer())
    return sess


def save(sess, saver, config, suffix = "", global_step = None):
    saver.save(sess, "./{}{}".format(config["output"], suffix), global_step)


# ===========================
# Helper function
# ===========================
def compute_action(sess, actor, obs, noise, test):
    if test:
        action = actor.predict(sess, np.reshape(obs, (1, actor.s_dim)))
    else:
        action = actor.predict(sess, np.reshape(obs, (1, actor.s_dim)))
        action += noise
    action = np.reshape(action, (actor.a_dim,))
    action = np.clip(action, -1, 1)
    return action

#def cur_gen(ss, params):
#    space = np.linspace(params[0], params[1], params[2])
#    idx = np.argmax(space>ss)
#    return space[idx]

def cur_gen(steps, x):
    vals = np.linspace(x[0], x[1], x[2])
    ss = np.linspace(0, steps, x[2])
    ss = ss + ss[1]
    for i, val in enumerate(vals):
        yield ss[i], val


# ===========================
#   Agent Training
# ===========================
def train(env, ddpg, actor, critic, **config):

    print("Noise: {} and {}".format(config["ou_sigma"], config["ou_theta"]))
    print("Actor learning rate {}".format(config["actor_lr"]))
    print("Critic learning rate {}".format(config["critic_lr"]))
    print("Minibatch size {}".format(config["minibatch_size"]))

    curriculums = []
    if config["curriculum"]:
        print("Following curriculum {}".format(config["curriculum"]))
        params = config["curriculum"].split("_")
        x = np.array(params[1:]).astype(np.float)
        c = {'var': params[0], 'gen': cur_gen(config["steps"], x)}
        curriculums.append(c)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    with tf.Session(graph=ddpg, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        saver = tf.train.Saver()

        # Check if a policy needs to be loaded
        sess = preload_policy(sess, saver, config)

        # Initialize target network weights
        actor.update_target_network(sess)
        critic.update_target_network(sess)

        # Initialize replay memory
        replay_buffer = ReplayBuffer(config)

        # Initialize constants for exploration noise
        ou_sigma = config["ou_sigma"]
        ou_theta = config["ou_theta"]
        ou_mu = 0
        trial_return = 0
        max_trial_return = 0

        obs_dim = actor.s_dim
        act_dim = actor.a_dim
        max_action = np.minimum(np.absolute(env.action_space.high),
                                np.absolute(env.action_space.low))

        obs = np.zeros(obs_dim)
        action = np.zeros(act_dim)
        noise = np.zeros(act_dim)

        tt = 0
        ss = 0
        terminal = 0
        ti = config["test_interval"]

        # start environment
        test = (ti>=0 and tt%(ti+1) == ti)
        env.set_test(test)
        for c in curriculums:
            c['ss'], val = next(c['gen'])
            d = {"action": "update", c['var']: val}
            env.reconfigure(d)
        obs = env.reset()

        # Main loop over steps or trials
        while (config["trials"] == 0 or tt < config["trials"]) and \
              (config["steps"]  == 0 or ss < config["steps"]):

            # Compute OU noise and action
            if not test:
                noise = ExplorationNoise.ou_noise(ou_theta, ou_mu, ou_sigma, noise, act_dim)

            action = compute_action(sess, actor, obs, noise, test) # from [-1; 1]

            # obtain observation of a state
            next_obs, reward, terminal, _ = env.step(action * max_action)

            # Add the transition to replay buffer
            if not test:
                replay_buffer.replay_buffer_add(obs, action, reward, terminal == 2, next_obs)

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if not test and replay_buffer.size() > config["rb_min_size"]:
                minibatch_size = config["minibatch_size"]
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(minibatch_size)

                # Calculate targets
                target_q = critic.predict_target(sess, s2_batch, actor.predict_target(sess, s2_batch))

                y_i = []
                for k in range(minibatch_size):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + config["gamma"] * target_q[k][0]) # target_q: list -> float

                # Update the critic given the targets
                critic.train(sess, s_batch, a_batch, np.reshape(y_i, (minibatch_size, 1)))

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(sess, s_batch)
                grad = critic.action_gradients(sess, s_batch, a_outs)[0]
                actor.train(sess, s_batch, grad)

                # Update target networks
                actor.update_target_network(sess)
                critic.update_target_network(sess)

            # Prepare next step
            obs = next_obs
            trial_return += reward

            # Logging performance at the end of the testing trial
            if terminal and test:
                msg = "{:>11} {:>11} {:>11.3f} {:>11.3f} {:>11}" \
                    .format(tt, ss, trial_return, max_trial_return, terminal)
                print("{}".format(msg))

            # Save NN if performance is better then before
            if terminal == 1 and config['save'] and trial_return > max_trial_return:
                max_trial_return = trial_return
                save(sess, saver, config, suffix="-best")

            if not test:
                ss = ss + 1
                for c in curriculums:
                    if ss > c['ss']:
                        c['ss'], val = next(c['gen'])
                        d = {"action": "update", c['var']: val}
                        env.reconfigure(d)

            if terminal:
                tt += 1
                test = (ti>=0 and tt%(ti+1) == ti)
                env.set_test(test)
                obs = env.reset()
                reward = 0
                terminal = 0
                trial_return = 0
                noise = np.zeros(actor.a_dim)


        # Save the last episode policy
        if config['save']:
            save(sess, saver, config, suffix="-last")


def start(env, **config):

    # Initialize the actor, critic and difference networks
    with tf.Graph().as_default() as ddpg:

        # setup random number generators for predicatbility
        random.seed(config['seed'])
        np.random.seed(random.randint(0, 1000))
        tf.set_random_seed(random.randint(0, 1000))
        env.seed(random.randint(0, 1000))

        obs_dim = env.observation_space.shape[-1]
        act_dim = env.action_space.shape[-1]

        actor = ActorNetwork(obs_dim, act_dim, 1, config["actor_lr"], config["tau"])
        critic = CriticNetwork(obs_dim, act_dim, config["critic_lr"], config["tau"],
                               actor.get_num_trainable_vars())

        if config["tensorboard"] == True:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            tf.summary.FileWriter(dir_path, ddpg)

        train(env, ddpg, actor, critic, **config)