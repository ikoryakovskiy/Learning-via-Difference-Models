import gym
from gym.core import Wrapper
import time
import csv
import os.path as osp
import json
from baselines.bench import Monitor

class MyMonitor(Monitor):
    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=(), report='test'):  
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if osp.isdir(filename):
                    filename = osp.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.f = open(filename, "wt")
            self.f.write('#%s\n'%json.dumps({"t_start": self.tstart, "gym_version": gym.__version__,
                "env_id": env.spec.id if env.spec else 'Unknown'}))
            self.logger = csv.DictWriter(self.f, fieldnames=('steps-reward-terminal-info',)+reset_keywords)
            self.logger.writeheader()

        self.reset_keywords = reset_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()
        self.test = False
        self.report = report
        env.report(report)

    def _step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.rewards.append(rew)
        self.total_steps += 1
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            line = "{:15d}{:15.5f}{:15d}{}".format(self.total_steps, eprew, done, info)
            epinfo = {"steps-reward-terminal-info": line}
            epinfo.update(self.current_reset_info)
            log_ok = (self.test and (self.report=='test' or self.report=='all')) or \
                     (not self.test and (self.report=='learn' or self.report=='all'))
            if self.logger and log_ok:
                self.logger.writerow(epinfo)
                self.f.flush()
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
        return (ob, rew, done, info)
    
    # own
    def set_test(self, test=False):
        self.test = test
        self.env.set_test(self.test)
        
    def _dict_to_string(self, rowdict):
        return (rowdict.get(key, self.restval) for key in self.fieldnames)