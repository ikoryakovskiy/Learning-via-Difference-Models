#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
from collections import deque
import pdb


class Evaluator(object):
    def __init__(self, max_action):
        DXL_XM430_210_MAX_TORQUE  = 3.49054054054
        DXL_XM430_210_MAX_CURRENT = 2.59575289575
        DXL_XM430_210_TORQUE_CONST = DXL_XM430_210_MAX_TORQUE/DXL_XM430_210_MAX_CURRENT
        self.DXL_XM430_210_GEARBOX_RATIO = 212.6
        self.DXL_XM430_210_TORQUE_CONST = DXL_XM430_210_TORQUE_CONST/self.DXL_XM430_210_GEARBOX_RATIO
        self.DXL_RESISTANCE = 4.6
        self.max_action = max_action


    def add_bonus(self, replay_buffer, rwForward=300, rwTime=-1.5,
                 how = 'walking'):
        items = how.split("_")
        if not items[0] == 'walking':
            return replay_buffer

        if len(items) > 1: rwForward = float(items[1])
        if len(items) > 2: rwTime = float(items[2])

        new_replay_buffer = deque()
        for e in replay_buffer.replay_buffer:
            r_old = e[2]
            r_new = r_old + rwForward*e[5] + rwTime
            #print(r_old, r_new, e[5])
            #pdb.set_trace()
            new_e = (e[0], e[1], r_new, e[3], e[4], e[5])
            new_replay_buffer.append(new_e)
        replay_buffer.replay_buffer = new_replay_buffer
        return replay_buffer


    def reassess(self, replay_buffer, rwForward=300, verify=False,
                   task = 'walking', knee_mode = "punish_and_continue"):
        new_replay_buffer = deque()
        for e in replay_buffer.replay_buffer:
            r_old = e[2]
            r_new = self.evaluateExperience(e[0], e[1]*self.max_action,
                                            e[3], e[4],
                                            e[5], rwForward,
                                            task, knee_mode)
            new_e = (e[0], e[1], r_new, e[3], e[4], e[5])
            new_replay_buffer.append(new_e)
            if verify:
                assert(r_old == r_new)
        replay_buffer.replay_buffer = new_replay_buffer
        return replay_buffer


    def evaluateExperience(self, s, a, t, s2, fw = 0, rwForward=300,
                           task = 'walking', knee_mode = "punish_and_continue"):
        """ When walking, time penalty and rwForward are added """
        r = 0
        rwFail = -75
        rwTime = -1.5
        rwBrokenKnee = -10
        rwWork = -2

        if t:
            r += rwFail
        else:
            if knee_mode == "punish_and_continue" and self.isKneeBroken(s2):
                r += rwBrokenKnee

        if task == 'walking':
            r += rwTime
            r += rwForward*fw;

        stepEnergy = self.getMotorWork(s, s2, a)
        r += rwWork*stepEnergy
        return r


    def getMotorWork(self, s, s2, a):

        motorWork = 0
        desiredFrequency = 30
        I , U = 0, 0 # Electrical work: P = U*I

        dof = len(s)//2
        for ii in range(1, dof):  # Start from 1 to ignore work by torso joint
            # We take the joint velocity as the average of the previous and the current velocity measurement
            omega = 0.5*(s[ii+dof] + s2[ii+dof])
            # We take the action that was executed the previous step.
            U = a[ii-1]
            I = (U - self.DXL_XM430_210_TORQUE_CONST*self.DXL_XM430_210_GEARBOX_RATIO*omega)/self.DXL_RESISTANCE;
            # Negative electrical work is not beneficial (no positive reward), but does not harm either.
            motorWork += max([0.0, U*I]) / desiredFrequency  # Divide power by frequency to get energy (work)
        return motorWork

    def isKneeBroken(self, s):
        if s[3] > 0 or s[4] > 0:
            return True
        else:
            return False



