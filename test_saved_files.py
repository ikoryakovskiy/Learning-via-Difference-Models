#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pickle
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats, integrate
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def main():
    fig = plt.figure()
    fig1 = plt.figure()
    dx1 = fig1.add_subplot(311)
    dx2 = fig1.add_subplot(312)
    dx3 = fig1.add_subplot(313)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    sns.set(color_codes=True)
    plot_points = 10000
    data_points = 10000
    angle = np.linspace(-math.pi, math.pi, num=data_points)
    vel = np.linspace(10, 10, num=data_points)
    act = np.linspace(2.8, 2.8, num=data_points)
    ss = deque()
    print act

    for i in range(data_points):
        data = (np.reshape((math.cos(angle[i]), math.sin(angle[i]), vel[i]), (3,)), np.reshape(act[i], (1,)))
        ss.append(data)
    time = np.reshape(np.linspace(1,10000,10000),(10000,1))
    print time.shape
    with open('state_space_data', 'w') as f:
        pickle.dump(ss, f, protocol=pickle.HIGHEST_PROTOCOL)

    transitions = deque()
    with open('saved_state_space_data_ideal') as f:
        transitions = pickle.load(f)
        f.close()

        # batch = xrange(10000)
        s = np.array([_[0] for _ in transitions])
        a = np.array([_[1] for _ in transitions])
        s2 = np.array([_[4] for _ in transitions])
        o = invert_state_obs(s)
        o_ideal_2 = invert_state_obs(s2)
        print o_ideal_2.shape
        b = np.concatenate((s,a,s2), axis=1)
        print b.shape
        np.savetxt('saved_state_space_data_ideal.csv',b,delimiter=',',newline='\n')
        print s2
        ax1.plot(time[1:plot_points,0], o_ideal_2[1:plot_points,1], 'b')
        ax2.plot(time[1:plot_points,0], o_ideal_2[1:plot_points,0], 'b')

    transitions = deque()
    with open('saved_state_space_data_diff') as f:
        transitions = pickle.load(f)
        f.close()
        s = np.array([_[0] for _ in transitions])
        a = np.array([_[1] for _ in transitions])
        s2 = np.array([_[4] for _ in transitions])
        sd = np.array([_[5] for _ in transitions])
        print(sd[1:100,2])
        o = invert_state_obs(s)
        o_diff_2 = invert_state_obs(s2)
        b = np.concatenate((s,a,s2), axis=1)
        print b.shape
        np.savetxt('saved_state_space_data_diff.csv',b,delimiter=',',newline='\n')
        # plt.plot(math.atan2(s2[1:plot_points, 1], s2[1:plot_points, 0]), 'r')
        ax1.plot(time[1:plot_points, 0], o_diff_2[1:plot_points, 1], 'r')
        ax2.plot(time[1:plot_points, 0], o_diff_2[1:plot_points, 0], 'r')
        ax1.errorbar(time[1:plot_points, 0],o_diff_2[1:plot_points,1] - o_ideal_2[1:plot_points,1], sd[1:plot_points, 2])
        ax1.plot(time[1:plot_points,0], o_diff_2[1:plot_points,1] - o_ideal_2[1:plot_points,1], 'r--')
        ax2.plot(time[1:plot_points,0], o_diff_2[1:plot_points,0] - o_ideal_2[1:plot_points,0], 'r--')
        ax2.set_ylim([-3, 3])
        # plt.plot(s2[:,2], 'r')


    transitions = deque()
    with open('saved_state_space_data_pert') as f:
        transitions = pickle.load(f)
        f.close()
        s = np.array([_[0] for _ in transitions])
        a = np.array([_[1] for _ in transitions])
        s2 = np.array([_[4] for _ in transitions])
        o = invert_state_obs(s)
        o_pert_2 = invert_state_obs(s2)
        b = np.concatenate((s,a,s2), axis=1)
        print b.shape
        np.savetxt('saved_state_space_data_pert.csv',b,delimiter=',',newline='\n')
        # plt.plot(math.atan2(s2[1:plot_points, 1], s2[1:plot_points, 0]), 'g')
        ax1.plot(time[1:plot_points, 0], o_pert_2[1:plot_points, 1], 'g')
        ax2.plot(time[1:plot_points, 0], o_pert_2[1:plot_points, 0], 'g')
        ax1.plot(time[1:plot_points,0], o_pert_2[1:plot_points,1] - o_ideal_2[1:plot_points,1], 'g--')
        ax2.plot(time[1:plot_points,0], o_pert_2[1:plot_points,0] - o_ideal_2[1:plot_points,0], 'g--')
        # plt.plot(s2[:,2], 'g')

    with open('saved_data-1') as f:
        transitions = pickle.load(f)
        f.close()

        # batch = xrange(10000)
        s = np.array([_[0] for _ in transitions])
        a = np.array([_[1] for _ in transitions])
        s2 = np.array([_[4] for _ in transitions])
        o = invert_state_obs(s)
        o_ideal_2 = invert_state_obs(s2)
        print o_ideal_2.shape
        b = np.concatenate((s,a,s2), axis=1)
        print b.shape
        np.savetxt('saved_state_data-1.csv',b,delimiter=',',newline='\n')
        # ax1.scatter(time[1:10000, 0], o2[1:10000, 1])
        # ax2.scatter(time[1:10000, 0], o2[1:10000, 0])

        # plt.show()

        sns.distplot(o[1:10000, 0], ax=dx1)
        sns.distplot(o[1:10000, 1], ax=dx2)
        sns.distplot(a[1:10000, 0], ax=dx3)

        plt.show()


def invert_state_obs(a):
    b = np.zeros((np.shape(a)[0],2))
    for i in range(np.shape(a)[0]):
        b[i][:] = np.array([math.atan2(a[i][1], a[i][0]), a[i][2]])
    return b

if __name__ == "__main__":
    main()