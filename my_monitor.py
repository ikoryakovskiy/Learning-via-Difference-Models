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
        self.done = 0
        self.step_info = {}             # info per every step
        self.episode_info = {}          # info at the episode end
        self.total_steps = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()
        self.test = False
        self.report = report
        try:
            self.env.report(self.report)
        except AttributeError:
            print("report method is not supported by the environment")

    def _reset(self, **kwargs):
        self.test = kwargs['test']
        return super(MyMonitor, self)._reset(**kwargs)

    def _step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, self.done, self.step_info = self.env.step(action)
        self.rewards.append(rew)
        if not self.test:
            self.total_steps += 1 # account for learning steps only
        if self.done:
            self.needs_reset = True
            self.episode_info = self.step_info
        return (ob, rew, self.done, self.step_info)


    # own
    def reconfigure(self, d=None):
        """ Reconfigure the environemnt using the dict """
        try:
            self.env.reconfigure(d)
        except AttributeError:
            print("reconfigure method is not supported by the environment")


    def get_latest_info(self):
        if self.step_info:
            return self.step_info
        else:
            return self.episode_info


    def log(self, more_info = None):
        eprew = sum(self.rewards)
        info = self.get_latest_info()
        if info:
            line = "{:10d}{:10.2f}{:10d}{}".format(self.total_steps, eprew, self.done, info)
        else:
            line = "{:10d}{:10.2f}{:10d}".format(self.total_steps, eprew, self.done)
        if more_info:
            line = "{} {}".format(line, more_info)
        epinfo = {"steps-reward-terminal-info": line}
        epinfo.update(self.current_reset_info)
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()


    def _dict_to_string(self, rowdict):
        return (rowdict.get(key, self.restval) for key in self.fieldnames)