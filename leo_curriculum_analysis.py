#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:49:02 2017

@author: divyam, ikoryakovskiy
"""
from grlgym.envs.grl import Leo
import time
import yaml
import tensorflow as tf
import numpy as np
import os.path
from replaybuffer_ddpg import ReplayBuffer
from assessment import Evaluator
from ExplorationNoise import ExplorationNoise
from actor import ActorNetwork
from critic import CriticNetwork
import random
import pdb
from running_mean_std import RunningMeanStd
import json
from ddpg import parse_args
from my_monitor import MyMonitor

# ===========================
# Policy saving and loading
# ===========================
def preload_policy(sess, saver, config):
    suffixes = ['', '-best', '-last']
    loaded = False
    if config["load_file"]:
        for sfx in suffixes:
            load_file = config["load_file"] + sfx
            path = os.path.dirname(os.path.abspath(__file__))
            load_file = "{}/{}".format(path, load_file)
            meta_file = "{}.meta".format(load_file)
            if os.path.isfile(meta_file):
                saver.restore(sess, load_file)
                print("Loaded NN from {}".format(meta_file))
                loaded = True
                break
        if not loaded:
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
def train(env, ddpg, actor, critic, balancing_graph, balancing_actor, **config):

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

    timestep = 0.03
    time = 0

    sim = []
    sim.append([time] + [0]*actor.a_dim)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    with tf.Session(graph=ddpg, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        for op in ddpg.get_operations():
            if "actorLayer1/W/Adam_1" in op.name:
                print(op)

        saver = tf.train.Saver()
        sess = preload_policy(sess, saver, config)

        with tf.Session(graph=balancing_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as balancing_sess:

            # also load balancing actor
            saver = tf.train.Saver()
            #saver.restore(balancing_sess, "leo_gait_analysis/ddpg-balancing-5000000-1010-mp1-last")
            saver.restore(balancing_sess, "leo_gait_analysis/ddpg-walking_after_balancing-25000000-1101-mp2-best")

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
                  (config["steps"]  == 0 or ss < config["steps"]):

                # Compute OU noise and action
                if not test:
                    noise = ExplorationNoise.ou_noise(ou_theta, ou_mu, ou_sigma, noise, act_dim)

                action = compute_action(sess, actor, obs[:o_dims], noise, test) # from [-1; 1]
                balancing_action = compute_action(balancing_sess, balancing_actor, obs[:o_dims], noise, test)

                time += timestep
                sim.append( [time] + abs(action-balancing_action).tolist() )


                # obtain observation of a state
                next_obs, reward, terminal, _ = env.step(action*max_action)
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

                # Save NN if performance is better then before
                if terminal and config['save'] and trial_return > max_trial_return:
    #                pdb.set_trace()
                    max_trial_return = trial_return
                    save(sess, saver, config, suffix="-best")

                if not test:
                    ss = ss + 1
                    for c in curriculums:
                        if ss > c['ss']:
                            c['ss'], val = next(c['gen'])
                            d = {"action": "update_{}".format(c['var']), c['var']: val}
                            env.reconfigure(d)

                if terminal:
    #                pdb.set_trace()
                    tt += 1
                    test = (ti>=0 and tt%(ti+1) == ti)
                    obs = env.reset(test=test)
                    obs = obs_normalize(obs, obs_rms, obs_range, o_dims, config["normalize_observations"])
                    reward = 0
                    terminal = 0
                    trial_return = 0
                    noise = np.zeros(actor.a_dim)


        # verify replay_buffer
        #evaluator.reassess(replay_buffer, verify=True, task = config['reassess_for'])

        # Save the last episode policy
        if config['save']:
            suffix="-last"
            save(sess, saver, config, suffix=suffix)
            if config["normalize_observations"]:
                with open(config["output"]+suffix+'.obs_rms', 'w') as f:
                    data = {'count': obs_rms.count, 'mean': obs_rms.mean.tolist(), 'std': obs_rms.std.tolist(), 'var': obs_rms.var.tolist()}
                    json.dump(data, f)

        np.savetxt('sim_leo_wb.csv', sim)

        replay_buffer.save()


def start(env, **config):

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

    with tf.Graph().as_default() as balancing_graph:
        balancing_actor = ActorNetwork(obs_dim, act_dim, 1, config)
        balancing_critic = CriticNetwork(obs_dim, act_dim, config,
                           balancing_actor.get_num_trainable_vars())

    print(actor.target_inputs.graph is tf.get_default_graph())
    print(balancing_actor.target_inputs.graph is tf.get_default_graph())
    print(balancing_actor.target_inputs.graph is actor.target_inputs.graph)

    train(env, ddpg, actor, critic, balancing_graph, balancing_actor, **config)


def cfg_run(**config):
    with open("{}.yaml".format(config['output']), 'w', encoding='utf8') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    del config['cores']
    if config['seed'] == None:
        config['seed'] = int.from_bytes(os.urandom(4), byteorder='big', signed=False) // 2
    run(**config)

def run(cfg, **config):
    # Create envs.
    if os.path.isfile(cfg):
        env = Leo(cfg)

    env = MyMonitor(env, config['output'])

    start_time = time.time()
    start(env=env, **config)
    print('total runtime: {}s'.format(time.time() - start_time))

    env.close()


if __name__ == '__main__':
    args = parse_args()

    env = 'leo'
    task = 'walking'
    #task = 'balancing'

    args['cfg'] = 'cfg/{}_{}_play.yaml'.format(env, task)
    args['seed'] = 1
    args['steps'] = 0
    args['trials'] = 1
    args['test_interval'] = 0
    args['normalize_observations'] = False
    args['normalize_returns'] = False
    args['batch_norm'] = True
    args['output'] = '{}_{}_play'.format(env, task)
    #args['load_file'] = 'leo_curriculum_analysis/ddpg-walking-30000000-1000-mp0-best'
    args['load_file'] = 'leo_gait_analysis/ddpg-walking_after_balancing-25000000-1101-mp2-best'

    # Run actual script.
    args['save'] = False
    cfg_run(**args)
