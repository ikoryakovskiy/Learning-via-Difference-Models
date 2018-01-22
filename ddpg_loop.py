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
from assessment import Evaluator
from ExplorationNoise import ExplorationNoise
from actor import ActorNetwork
from critic import CriticNetwork
from cl_network import CurriculumNetwork
import random
from running_mean_std import RunningMeanStd
import json
import pdb


# ===========================
# Policy saving and loading
# ===========================
def load_policy(sess, config):
    suffixes = ['', '-best', '-last']
    loaded = False
    if config["load_file"]:
        for sfx in suffixes:
            load_file = config["load_file"] + sfx
            path = os.path.dirname(os.path.abspath(__file__))
            load_file = "{}/{}".format(path, load_file)
            meta_file = "{}.meta".format(load_file)
            if os.path.isfile(meta_file):
                var_all = tf.trainable_variables()
                var_this = [v for v in var_all if not 'curriculum' in v.name]
                saver = tf.train.Saver(var_this)
                saver.restore(sess, load_file)
                print("Loaded NN from {}".format(meta_file))
                loaded = True
                break
        if not loaded:
            print("Not a valid path")
    return sess


def save_policy(sess, config, suffix = "", global_step = None):
    if config["output"]:
        var_all = tf.trainable_variables()
        var_this = [v for v in var_all if not 'curriculum' in v.name]
        saver = tf.train.Saver(var_this)
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

def cur_gen(steps, x):
    if x.size == 3:
        vals = np.linspace(x[0], x[1], x[2])
        ss = np.linspace(0, steps, x[2])
        ss = ss + ss[1]
        for i, val in enumerate(vals):
            yield ss[i], val
    else:
        # a single number which sets a certain life-time value
        while True:
            yield steps, x[0]

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std

def obs_normalize(obs, obs_rms, obs_range, o_dims, normalize_observations):
    obsx = obs[np.newaxis, :o_dims]
    if normalize_observations:
        obs_rms.update(obsx)
    #pdb.set_trace()
    obs[:o_dims] = np.clip(normalize(obsx, obs_rms) , obs_range[0], obs_range[1])
    return obs

# ===========================
#   Agent Training
# ===========================
def train(env, ddpg_graph, actor, critic, cl_nn = None, pt = None, cl_mode=None, **config):

    print("Noise: {} and {}".format(config["ou_sigma"], config["ou_theta"]))
    print("Actor learning rate {}".format(config["actor_lr"]))
    print("Critic learning rate {}".format(config["critic_lr"]))
    print("Minibatch size {}".format(config["minibatch_size"]))

    curriculums = []
    if config["curriculum"]:
        print("Following curriculum {}".format(config["curriculum"]))
        items = config["curriculum"].split(";")
        for item in items:
            params = item.split("_")
            x = np.array(params[1:]).astype(np.float)
            c = {'var': params[0], 'gen': cur_gen(config["steps"], x)}
            curriculums.append(c)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    with tf.Session(graph=ddpg_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # random initialization of variables
        sess.run(tf.global_variables_initializer())

        # load curriculum neural network weights (provided parametes have priority)
        if cl_nn:
            sess = cl_nn.load(sess, config["cl_load"])

        # Check if a policy needs to be loaded
        sess = load_policy(sess, config)

        # Initialize target network weights
        actor.update_target_network(sess)
        critic.update_target_network(sess)

        # Initialize replay memory
        o_dims=env.observation_space.shape[-1]
        replay_buffer = ReplayBuffer(config, o_dims=o_dims)

        # Observation normalization.
        obs_range = [env.observation_space.low, env.observation_space.high]
        #pdb.set_trace()
        #obs_range = [-5, 5]
        if config["normalize_observations"]:
            obs_rms = RunningMeanStd(shape=env.observation_space.shape)
        else:
            obs_rms = None

        # decide mode
        cl_mode_new = None
        if config['cl_on']:
            v = pt.flatten()
            cl_mode_new = cl_nn.predict(sess, v)
            #cl_nn.get_params(sess)

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
        terminal_info = None
        prev_falls = 0
        ti = config["test_interval"]
        test_returns = []

        # rewarding object if rewards in replay buffer are to be recalculated
        replay_buffer.load()
        if config['reassess_for']:
            print('Reassessing replay buffer for {}'.format(config['reassess_for']))
            evaluator = Evaluator(max_action)
            #pdb.set_trace()
            replay_buffer = evaluator.add_bonus(replay_buffer, how = config['reassess_for'])

        # start environment
        for c in curriculums:
            c['ss'], val = next(c['gen'])
            d = {"action": "update_{}".format(c['var']), c['var']: val}
            env.reconfigure(d)
        test = (ti>=0 and tt%(ti+1) == ti)
        obs = env.reset(test=test)
        obs = obs_normalize(obs, obs_rms, obs_range, o_dims, config["normalize_observations"])

        # Main loop over steps or trials
        while (config["trials"] == 0 or tt < config["trials"]) and \
              (config["steps"]  == 0 or ss < config["steps"]) and \
              (cl_mode_new == cl_mode or not config['cl_on']):

            # Compute OU noise and action
            if not test:
                noise = ExplorationNoise.ou_noise(ou_theta, ou_mu, ou_sigma, noise, act_dim)

            action = compute_action(sess, actor, obs[:o_dims], noise, test) # from [-1; 1]

            # obtain observation of a state
            next_obs, reward, terminal, info = env.step(action*max_action)
            next_obs = obs_normalize(next_obs, obs_rms, obs_range, o_dims, config["normalize_observations"])

            reward *= config['reward_scale']
            #pdb.set_trace()

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

            # Render
            if config["render"]:
                still_open = env.render("human")
                if still_open==False:
                    break

            # Prepare next step
            obs = next_obs
            trial_return += reward

            # Logging performance at the end of the testing trial
            if terminal and test:
                msg = "{:>11} {:>11} {:>11.3f} {:>11.3f} {:>11}" \
                    .format(tt, ss, trial_return, max_trial_return, terminal)
                print("{}".format(msg))

                # update PerformanceTracker
                if cl_nn:
                    s = info.split()
                    # convert number of falls into a relative value
                    damage = (float(s[1]) - prev_falls) / ti
                    pt.add([trial_return, float(s[0]), damage]) # return, duration, damage
                    v = pt.flatten()
                    cl_mode_new = cl_nn.predict(sess, v)
                    prev_falls = float(s[1])
                test_returns.append(trial_return)

            # Save NN if performance is better then before
            if terminal and config['save'] and trial_return > max_trial_return:
                max_trial_return = trial_return
                save_policy(sess, config, suffix="-best")

            if not test:
                ss = ss + 1
                for c in curriculums:
                    if ss > c['ss']:
                        c['ss'], val = next(c['gen'])
                        d = {"action": "update_{}".format(c['var']), c['var']: val}
                        env.reconfigure(d)

            if terminal:
                tt += 1
                test = (ti>=0 and tt%(ti+1) == ti)
                obs = env.reset(test=test)
                obs = obs_normalize(obs, obs_rms, obs_range, o_dims, config["normalize_observations"])
                reward = 0
                terminal = 0
                trial_return = 0
                noise = np.zeros(actor.a_dim)
                terminal_info = info

        # verify replay_buffer
        #evaluator.reassess(replay_buffer, verify=True, task = config['reassess_for'])

        # Save the last episode policy
        if config['save']:
            suffix="-last"
            save_policy(sess, config, suffix=suffix)
            if config["normalize_observations"]:
                with open(config["output"]+suffix+'.obs_rms', 'w') as f:
                    data = {'count': obs_rms.count, 'mean': obs_rms.mean.tolist(), 'std': obs_rms.std.tolist(), 'var': obs_rms.var.tolist()}
                    json.dump(data, f)

        replay_buffer.save()

        # save curriculum network
        if cl_nn:
            cl_nn.save(sess, config["cl_save"])

        # extract damage from the last step
        damage = 0
        if terminal_info:
            s = terminal_info.split()
            damage = float(s[1])

    return (test_returns, damage, ss, cl_mode_new)


def start(env, pt=None, cl_mode=None, **config):

    # Initialize the actor, critic and difference networks
    with tf.Graph().as_default() as ddpg:

        # setup random number generators for predicatbility
        print("Random seed ", config['seed'])
        random.seed(config['seed'])
        np.random.seed(random.randint(0, 10000))
        tf.set_random_seed(random.randint(0, 10000))
        env.seed(random.randint(0, 10000))

        obs_dim = env.observation_space.shape[-1]
        act_dim = env.action_space.shape[-1]

        actor = ActorNetwork(obs_dim, act_dim, 1, config)
        critic = CriticNetwork(obs_dim, act_dim, config,
                               actor.get_num_trainable_vars())

        if config["tensorboard"] == True:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            tf.summary.FileWriter(dir_path, ddpg)

        # create curriculum switching network
        cl_nn = None
        if config["cl_on"]:
            cl_nn = CurriculumNetwork(pt.get_v_size(), 1, config)

    return train(env, ddpg, actor, critic, cl_nn, pt, cl_mode, **config)
