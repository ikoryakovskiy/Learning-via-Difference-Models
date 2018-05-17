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
import pickle

# ===========================
# Policy saving and loading
# ===========================
def preload_policy(sess, config):
    suffixes = ['', '-best', '-last']
    loaded = False
    if config["load_file"]:
        for sfx in suffixes:
            load_file = config["load_file"] + sfx
            path = os.path.dirname(os.path.abspath(__file__))
            load_file = "{}/{}".format(path, load_file)
            meta_file = "{}.meta".format(load_file)
            if os.path.isfile(meta_file):
                var_all = tf.global_variables() #tf.trainable_variables()
                var_this = [v for v in var_all if not 'curriculum' in v.name]
                saver = tf.train.Saver(var_this)
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


def save_policy(sess, config, suffix = "", global_step = None):
    if config["output"]:
        var_all = tf.global_variables() #tf.trainable_variables()
        var_this = [v for v in var_all if not 'curriculum' in v.name]
        saver = tf.train.Saver(var_this)
        saver.save(sess, "./{}{}".format(config["output"], suffix), global_step)


def dump_pkl_csv(fname, data):
    if len(data):
        if '.pkl' in fname:
            with open(fname,'wb') as f:
                pickle.dump(np.array(data), f)
        else:
            np.savetxt(fname+'.csv', data)

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
#   Compare Agents
# ===========================
def compare(env, ddpg, actor, critic, compare_with_graph, compare_with_actor,
            cl_nn = None, pt = None, cl_mode=None, **config):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    with tf.Session(graph=compare_with_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as compare_with_sess:
        # also load balancing actor
        saver = tf.train.Saver()
        saver.restore(compare_with_sess, config["compare_with"])
        train(env, ddpg, actor, critic, cl_nn=cl_nn, pt=pt, cl_mode=cl_mode,
              compare_with_sess=compare_with_sess, compare_with_actor=compare_with_actor, **config)

# ===========================
#   Agent Training
# ===========================
def train(env, ddpg_graph, actor, critic, cl_nn = None, pt = None, cl_mode=None, compare_with_sess=None, compare_with_actor=None, **config):

    print('train: ' + config['output'] + ' started!')
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

        # Check if a policy needs to be loaded
        sess = preload_policy(sess, config)

        # Initialize target network weights
        actor.update_target_network(sess)
        critic.update_target_network(sess)

        # Load curriculum neural network weights (provided parametes have priority)
        if cl_nn:
            sess = cl_nn.load(sess, config["cl_load"])

        # Initialize replay memory
        o_dims=env.observation_space.shape[-1]
        replay_buffer = ReplayBuffer(config, o_dims=o_dims)

        # Observation normalization.
        obs_range = [env.observation_space.low, env.observation_space.high]
        if config["normalize_observations"]:
            obs_rms = RunningMeanStd(shape=env.observation_space.shape)
        else:
            obs_rms = None

        # decide mode
        if cl_nn:
            v = pt.flatten()
            cl_mode_new, cl_threshold = cl_nn.predict(sess, v)
            #cl_threshold = pt.denormalize(cl_threshold)
        else:
            cl_mode_new = cl_mode
            cl_threshold = None

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
        ss_all = 0
        terminal = 0
        reach_timeout_num = 0
        more_info = None
        ss_acc, td_acc, l2_reg_acc, action_grad_acc, actor_grad_acc = 0,0,0,0,0
        prev_l2_reg = critic.l2_reg_(sess)
        ti = config["test_interval"]
        test_returns = []
        avg_test_return = config['reach_return']

        # rewarding object if rewards in replay buffer are to be recalculated
        replay_buffer.load()
        if config['reassess_for']:
            print('Reassessing replay buffer for {}'.format(config['reassess_for']))
            evaluator = Evaluator(max_action)
            #pdb.set_trace()
            replay_buffer = evaluator.add_bonus(replay_buffer, how = config['reassess_for'])

        # Export trajectory
        if config['trajectory']:
            trajectory = []
            if config["compare_with"]:
                actor_sim = []

        # start environment
        for c in curriculums:
            c['ss'], val = next(c['gen'])
            d = {"action": "update_{}".format(c['var']), c['var']: val}
            env.reconfigure(d)
        test = (ti>=0 and tt%(ti+1) == ti)
        obs = env.reset(test=test)
        obs = obs_normalize(obs, obs_rms, obs_range, o_dims, config["normalize_observations"])

        # Export environment state
        if cl_nn:
            more_info = ''.join('{:10.2f}'.format(indi) for indi in [-100, -100, -100])
            more_info += ''.join('{:10.2f}'.format(vvv) for vv in v[0] for vvv in vv)
            more_info += ''.join('{:10.2f}'.format(th) for th in cl_threshold)
        env.log(more_info if cl_threshold is not None else '')

        # Main loop over steps or trials
        # Finish when trials finish
        # or Finish when steps finish
        # or Finishe when new mode in curriculum is switched
        # or Finish when certain return is reached
        # of Finish if trial happend to be longer then config['reach_balance'] twice in a row
        while (config["trials"] == 0 or tt < config["trials"]) and \
              (config["steps"]  == 0 or ss < config["steps"]) and \
              (not cl_nn or cl_mode_new == cl_mode) and \
              (not config['reach_return'] or avg_test_return <= config['reach_return']) and \
              (not config['reach_timeout'] or (config['reach_timeout'] > 0 and reach_timeout_num < config['reach_timeout_num'])):

            # Compute OU noise and action
            if not test:
                noise = ExplorationNoise.ou_noise(ou_theta, ou_mu, ou_sigma, noise, act_dim)

            action = compute_action(sess, actor, obs[:o_dims], noise, test) # from [-1; 1]

            # obtain observation of a state
            next_obs, reward, terminal, info = env.step(action*max_action)
            #print('Forward promotion: ' + str(next_obs[-1]))
            #print('Reward: ' + str(reward))
            next_obs = obs_normalize(next_obs, obs_rms, obs_range, o_dims, config["normalize_observations"])

            reward *= config['reward_scale']

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

                if config['perf_td_error']:
                    q_i = critic.predict_target(sess, s_batch, a_batch)
                    td_acc += np.sum(np.abs(q_i-np.reshape(y_i,newshape=(minibatch_size,1))))

                # Update the critic given the targets
                if config['perf_l2_reg']:
                    _, _, l2_reg = critic.train_(sess, s_batch, a_batch, np.reshape(y_i, (minibatch_size, 1)))
                    l2_reg_acc += (l2_reg - prev_l2_reg)
                    prev_l2_reg = l2_reg
                else:
                    critic.train(sess, s_batch, a_batch, np.reshape(y_i, (minibatch_size, 1)))

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(sess, s_batch)
                grad = critic.action_gradients(sess, s_batch, a_outs)[0]
                if config['perf_action_grad']:
                    action_grad_acc += np.linalg.norm(grad, ord=2)

                if config['perf_actor_grad']:
                    _, actor_grad = actor.train_(sess, s_batch, grad)
                    for ag in actor_grad:
                        actor_grad_acc += np.linalg.norm(ag, ord=2)
                else:
                    actor.train(sess, s_batch, grad)

                ss_acc += 1

                # Update target networks
                actor.update_target_network(sess)
                critic.update_target_network(sess)

            # Render
            if config["render"]:
                still_open = env.render("human")
                if still_open==False:
                    break

            # Record trajectory
            # Note that it exports all training and testing episodes
            if config['trajectory']:
                real_time = ss_all * config['env_timestep']
                trajectory.append([real_time] + obs[:o_dims].tolist() + (action*max_action).tolist() + next_obs[:o_dims].tolist() + [reward] + [terminal]) # + [info])
                if config["compare_with"]:
                    compare_with_action = compute_action(compare_with_sess, compare_with_actor, obs[:o_dims], noise, test)
                    actor_sim.append( [real_time] + obs[:o_dims].tolist() + (compare_with_action*max_action).tolist())

            # Prepare next step
            obs = next_obs
            trial_return += reward

            # Logging performance at the end of the testing trial
            if terminal and test:

                # NN performance indicators
                more_info = ""
                s = info.split()
                norm_duration = float(s[0]) / config["env_timeout"]
                td_per_step = td_acc/ss_acc if ss_acc > 0 else 0
                norm_td_error = td_per_step / config["env_td_error_scale"]
                norm_complexity = l2_reg_acc/ss_acc if ss_acc > 0 else 0
                indicators = [norm_duration, norm_td_error, norm_complexity]
                more_info += ''.join('{:14.8f}'.format(indi) for indi in indicators)
                if cl_nn:
                    # update PerformanceTracker
                    pt.add(indicators) # return, duration, damage
                    v = pt.flatten()
                    cl_mode_new, cl_threshold = cl_nn.predict(sess, v)
                    more_info += ''.join('{:14.8f}'.format(vvv) for vv in v[0] for vvv in vv)
                    more_info += ''.join('{:14.8f}'.format(th) for th in cl_threshold)
                ss_acc, td_acc, l2_reg_acc, action_grad_acc, actor_grad_acc = 0,0,0,0,0
                # report
                env.log(more_info)

                # check if performance is satisfactory
                test_returns.append(trial_return)
                avg_test_return = np.mean(test_returns[max([0, len(test_returns)-10]):])
                if float(info.split()[0]) > config['reach_timeout']:
                    reach_timeout_num += 1
                else:
                    reach_timeout_num = 0

                if not config['mp_debug']:
                    msg = "{:>10} {:>10} {:>10.3f} {:>10}" \
                        .format(tt, ss, trial_return, terminal)
                    print("{}".format(msg))


            # Save NN if performance is better then before
            if terminal and config['save'] and trial_return > max_trial_return:
                max_trial_return = trial_return
                save_policy(sess, config, suffix="-best")

            if not test:
                ss += 1
                for c in curriculums:
                    if ss > c['ss']:
                        c['ss'], val = next(c['gen'])
                        d = {"action": "update_{}".format(c['var']), c['var']: val}
                        env.reconfigure(d)
            ss_all += 1

            if terminal:
                tt += 1
                test = (ti>=0 and tt%(ti+1) == ti)
                obs = env.reset(test=test)
                obs = obs_normalize(obs, obs_rms, obs_range, o_dims, config["normalize_observations"])
                reward = 0
                terminal = 0
                trial_return = 0
                noise = np.zeros(actor.a_dim)

        # Export final performance, but when curriculum is not used or terminated
        # not due to the curriculum swithch.
        # Becasue data is always exported when curriculum is switched over.
        if (not cl_nn or cl_mode_new == cl_mode):
            env.log(more_info)

        # Export trajectory
        if config['trajectory']:
            dump_pkl_csv(config['trajectory'], trajectory)
            if config["compare_with"]:
                dump_pkl_csv(config['trajectory']+'_sim', actor_sim)

        # verify replay_buffer
        #evaluator.reassess(replay_buffer, verify=True, task = config['reassess_for'])
        print('train: ' + config['output'] + ' finished!')

        # Save the last episode policy
        if config['save']:
            suffix="-last"
            save_policy(sess, config, suffix=suffix)
            #save_policy(sess, saver, config, suffix=suffix)
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
        info = env.get_latest_info()
        if info:
            s = info.split()
            damage = float(s[1])

    print('train: ' + config['output'] + ' returning ' + '{} {} {} {}'.format(avg_test_return, damage, ss, cl_mode_new))

    return (avg_test_return, damage, ss, cl_mode_new)


def start(env, pt=None, cl_mode=None, **config):

    # block warnings from tf.saver if needed
    if config['mp_debug']:
        tf.logging.set_verbosity(tf.logging.ERROR)

    # Initialize the actor, critic and difference networks
    with tf.Graph().as_default() as ddpg:

        # setup random number generators for predicatbility
        print("Random seed ", config['seed'])
        random.seed(config['seed'])
        np.random.seed(random.randint(0, 1000000))
        tf.set_random_seed(random.randint(0, 1000000))
        env.seed(random.randint(0, 1000000))
        print("Random seed verification (numpy) ", np.random.randint(10000))

        obs_dim = env.observation_space.shape[-1]
        act_dim = env.action_space.shape[-1]

        actor = ActorNetwork(obs_dim, act_dim, 1, config)
        critic = CriticNetwork(obs_dim, act_dim, config,
                               actor.get_num_trainable_vars())

        if config["tensorboard"] == True:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            tf.summary.FileWriter(dir_path, ddpg)

        # create curriculum switching network
        if not config["cl_structure"] or not config["cl_stages"]:
            cl_nn = None
        else:
            cl_nn = CurriculumNetwork(pt.get_v_size(), config, cl_mode)

    if config["compare_with"]:
        with tf.Graph().as_default() as compare_with_graph:
            compare_with_actor = ActorNetwork(obs_dim, act_dim, 1, config)
            CriticNetwork(obs_dim, act_dim, config, compare_with_actor.get_num_trainable_vars())
        print(actor.target_inputs.graph is tf.get_default_graph())
        print(compare_with_actor.target_inputs.graph is tf.get_default_graph())
        print(compare_with_actor.target_inputs.graph is actor.target_inputs.graph)
        return compare(env, ddpg, actor, critic, compare_with_graph, compare_with_actor, cl_nn, pt, cl_mode,
                      **config)

    return train(env, ddpg, actor, critic, cl_nn, pt, cl_mode, **config)

