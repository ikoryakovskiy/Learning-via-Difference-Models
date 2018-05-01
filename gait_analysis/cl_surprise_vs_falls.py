"""RL data container."""

import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib import cm

from joblib import Parallel, delayed
from collections import OrderedDict

sys.path.append('/home/ivan/work/scripts/py')
from my_plot.plot import export_plot
from my_csv.utils import get_header_size, listsubsample
from my_stat.stat import mean_confidence_dd
from matplotlib.patches import Polygon
import pickle
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.close('all')

palette = ['#C0C0C0', '#4355bf', '#51843b', '#ffb171', '#408da9', '#408da9']

subfigprop3 = {
        'figsize': (4.2, 4.2),
        'dpi': 80,
        'facecolor': 'w',
        'edgecolor': 'k'
}

def compsim(data, arg):

    sm_beg, sm_end, ex_beg, ex_end = arg

    dist0 = np.zeros((sm_end-sm_beg,))
    for sample in range(sm_beg, sm_end):
        dist1 = np.zeros((ex_end-ex_beg+1,))
        for ex in range(ex_beg, ex_end+1):
            dist1[ex-ex_beg] = np.linalg.norm(data[ex, :] - data[sample, :])
        dist0[sample-sm_beg] = np.min(dist1)

    return [np.min(dist0), np.mean(dist0), np.max(dist0)]


def single(data, change, sm_batch_sz=30):
    data_sz = data.shape[0]
    # sm is a sliding window, current samples to considere
    # ex is a stretching window that stretches from 0 till be begining of the sliding window minus gap_sz

    gap_sz = 1 # increase if skip is decreased
    ex_beg = 0
    ex_end = 0

    mp_sm_beg = ex_end + gap_sz
    mp_sm_end = data_sz
    mp_sm_begs = list(range(mp_sm_beg, mp_sm_end-sm_batch_sz))
    mp_sm_ends = [beg+sm_batch_sz for beg in mp_sm_begs]

    mp_ex_begs = [ex_beg]*len(mp_sm_begs)
    mp_ex_ends = [beg-gap_sz for beg in mp_sm_begs]

    #timesteps = (np.array(mp_sm_begs) + np.array(mp_sm_ends)) // 2
    timesteps = np.array(mp_sm_begs) - gap_sz

#    mp_ex_begs = np.array(mp_ex_begs)
#    mp_ex_ends = np.array(mp_ex_ends)
#    mp_sm_begs = np.array(mp_sm_begs)
#    mp_sm_ends = np.array(mp_sm_ends)
#    for ch in change:
#        mp_sm_ends[range(ch-sm_batch_sz, ch)] = ch+1
#        #mp_sm_begs[range(ch-sm_batch_sz, ch)] = ch-sm_batch_sz+1

    mp_args = zip(mp_sm_begs, mp_sm_ends, mp_ex_begs, mp_ex_ends)

    similarity = Parallel(n_jobs=-1, verbose=10, backend="multiprocessing")(
                 delayed(compsim)(data, arg) for arg in mp_args)
    similarity = np.array(similarity)

#    # remove sm_batch_sz elements before change
#    rows_to_delete = []
#    for ch in change:
#        rows_to_delete += range(ch-sm_batch_sz, ch)
#
#    similarity = np.delete(similarity, rows_to_delete, axis=0)
#    timesteps = np.delete(timesteps, rows_to_delete, axis=0)

    return (timesteps, similarity)


def model_states():
    leo_state = OrderedDict([
            ('TorsoAngle',       1),
            ('LeftHipAngle',     2),
            ('RightHipAngle',    3),
            ('LeftKneeAngle',    4),
            ('RightKneeAngle',   5),
            ('LeftAnkleAngle',   6),
            ('RightAnkleAngle',  7),

            ('TorsoAngleRate',      8),
            ('LeftHipAngleRate',    9),
            ('RightHipAngleRate',   10),
            ('LeftKneeAngleRate',   11),
            ('RightKneeAngleRate',  12),
            ('LeftAnkleAngleRate',  13),
            ('RightAnkleAngleRate', 14),
    ])
    offset = 1+8
    hopper_state = OrderedDict([
            ('VxAngleRate',     offset-5),
            ('VzAngleRate',     offset-3),
            ('TorsoAngle',      offset-1),
            ('HipAngle',        offset+0),
            ('HipAngleRate',    offset+1),
            ('KneeAngle',       offset+2),
            ('KneeAngleRate',   offset+3),
            ('AnkleAngle',      offset+4),
            ('AnkleAngleRate',  offset+5),
    ])
    halfcheetah_state = OrderedDict([
            ('VxAngleRate',         offset-5),
            ('VzAngleRate',         offset-3),
            ('TorsoAngle',          offset-1),
            ('BackHipAngle',        offset+0),
            ('BackHipAngleRate',    offset+1),
            ('BackKneeAngle',       offset+2),
            ('BackKneeAngleRate',   offset+3),
            ('BackAnkleAngle',      offset+4),
            ('BackAnkleAngleRate',  offset+5),
            ('FrontHipAngle',       offset+6),
            ('FrontHipAngleRate',   offset+7),
            ('FrontKneeAngle',      offset+8),
            ('FrontKneeAngleRate',  offset+9),
            ('FrontAnkleAngle',     offset+10),
            ('FrontAnkleAngleRate', offset+11),
    ])
    walker2d_state = OrderedDict([
            ('VxAngleRate',          offset-5),
            ('VzAngleRate',          offset-3),
            ('TorsoAngle',           offset-1),
            ('RightHipAngle',        offset+0),
            ('RightHipAngleRate',    offset+1),
            ('RightKneeAngle',       offset+2),
            ('RightKneeAngleRate',   offset+3),
            ('RightAnkleAngle',      offset+4),
            ('RightAnkleAngleRate',  offset+5),
            ('LeftHipAngle',         offset+6),
            ('LeftHipAngleRate',     offset+7),
            ('LeftKneeAngle',        offset+8),
            ('LeftKneeAngleRate',    offset+9),
            ('LeftAnkleAngle',       offset+10),
            ('LeftAnkleAngleRate',   offset+11),
    ])
    return leo_state, hopper_state, halfcheetah_state, walker2d_state

def split_state(state, modif=''):
    angles = []
    anglerates = []
    for key in state:
        if 'AngleRate' in key and modif in key:
            anglerates.append(key)
        elif 'Angle' in key and modif in key:
            angles.append(key)
    return (angles, anglerates)

def extract_state(data, state, pick=None):
    columns = []
    if pick:
        for key in pick:
            columns.append(state[key])
    else:
        for key in state:
            columns.append(state[key])
    ret = data[:, columns]
    return ret

def main():
    skip = 500 #100
    kwargs = {
            "use_cols": ['mean'],
            "load": 1,
            "use_state":"all",
            "xlog": False,
            "ylog": False,
            "hours2plot": 2.0,
            "naming": ['walking', 'curriculum'],
            "colors": (palette[1], palette[3])
            }

    leo_state, hopper_state, halfcheetah_state, walker2d_state = model_states()

    ###
    kwargs["surprise_threshold"] = 2.0
    legends = ['I (w)', 'II (nn.rb)']
    w =  ['{path}/ddpg-exp1_two_stage_{env}_ga_w-g0001-mp{mp}-02_walking.pkl']
    bw = [
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload_rbload-g0001-mp{mp}-01_balancing.pkl',
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload_rbload-g0001-mp{mp}-02_walking.pkl',
         ]
    plot_single_system('leo', w, bw, leo_state, skip, legends, **kwargs)

    ###
    skip = 1000
    ###
    kwargs["surprise_threshold"] = 1.0
    legends = ['I (w)', 'II (nn)']
    w =  ['{path}/ddpg-exp1_two_stage_{env}_ga_w-g0001-mp{mp}-02_walking.pkl']
    bw = [
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload-g0001-mp{mp}-01_balancing.pkl',
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload-g0001-mp{mp}-02_walking.pkl',
         ]
    plot_single_system('hopper', w, bw, hopper_state, skip, legends, **kwargs)

    ###
    kwargs["surprise_threshold"] = 1.0
    w =  ['{path}/ddpg-exp1_two_stage_{env}_ga_w-g0001-mp{mp}-02_walking.pkl']
    bw = [
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload-g0001-mp{mp}-01_balancing.pkl',
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload-g0001-mp{mp}-02_walking.pkl',
         ]
    plot_single_system('halfcheetah', w, bw, halfcheetah_state, skip, legends, **kwargs)

    ###
    kwargs["surprise_threshold"] = 1.0
    w =  ['{path}/ddpg-exp1_two_stage_{env}_ga_w-g0001-mp{mp}-02_walking.pkl']
    bw = [
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload-g0001-mp{mp}-01_balancing.pkl',
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload-g0001-mp{mp}-02_walking.pkl',
         ]
    plot_single_system('walker2d', w, bw, walker2d_state, skip, legends, **kwargs)


def plot_single_system(env, w, bw, state, skip, legends, **kwargs):
    subplots = 2
    path = 'learning_trjectories'
    mps = range(1,6)

    fig, axarr = plt.subplots(subplots, sharex=True, **subfigprop3)

    norm_val = min_max(path, env, state, skip=skip, ffiles=(w,bw), mps=mps, **kwargs)

    dd1, _ = load_walking(path, env, state, skip=skip, files=w, mps=mps, norm_val=norm_val, **kwargs)
    dd2, timing = load_walking(path, env, state, skip=skip, files=bw, mps=mps, norm_val=norm_val, **kwargs)

    together = True
    if together:
        plot_together((dd1, dd2), timing, axarr, legends, **kwargs)
    else:
        ax0, ax1 = [], []
        a0, a1 = plot_separately(dd1, axarr[0], **kwargs)
        ax0.append(a0)
        ax1.append(a1)
        a0, a1 = plot_separately(dd2, axarr[1], **kwargs)
        ax0.append(a0)
        ax1.append(a1)

        ax0_lim = ax0[0].get_ylim()
        ax1_lim = ax1[0].get_ylim()
        for a0, a1 in zip(ax0, ax1):
            lim = a0.get_ylim()
            ax0_lim = [0, max(ax0_lim[1], lim[1])]
            lim = a1.get_ylim()
            ax1_lim = [0, max(ax1_lim[1], lim[1])]

        for a0, a1 in zip(ax0, ax1):
            a0.set_ylim(ax0_lim[0], ax0_lim[1])
            a1.set_ylim(ax1_lim[0], ax1_lim[1])

    if timing:
        for t in timing[1:-1]:
            #ax0[-1].axvspan(t[0], t[1], facecolor='0.8', zorder=0)
            tt = np.mean(t)
            for ax in axarr:
                ax.plot((tt, tt), ax.get_ylim(), 'r-.', zorder=0)

    axarr[1].set_xlabel('Time (h)')
    plt.subplots_adjust(left=0.15, bottom=0.13, right=0.9, top=0.98, wspace=0.0, hspace=0.1)
    export_plot('surprise_fall_rate_{}'.format(env))#, export_latex=False)

    # surprise plot
    surprise_plot((dd1, dd2), env, **kwargs)

    plt.show()



def min_max(path, env, state, skip=1000, ffiles=(), mps=[0], **kwargs):
    load = kwargs['load']

    angles, anglerates = split_state(state)
    if kwargs['use_state'] == 'all':
        pick = angles + anglerates
    elif kwargs['use_state'] == 'angles':
        pick = angles
    elif kwargs['use_state'] == 'anglerates':
        pick = anglerates

    # loding transitions
    if load == 0:
        all_states = []
        for files in ffiles:
            for mp in mps:
                data = []
                for fname in files:
                    fname_full = fname.format(path=path, env=env, mp=mp)
                    if os.path.isfile(fname_full):
                        with open(fname_full, 'rb') as f:
                            d = pickle.load(f)
                            data.append(d)
                data = np.concatenate(data)
                states = extract_state(data, state, pick)[::skip, :]
                all_states.append(states)

        all_states = np.concatenate(all_states)
        m = np.mean(all_states, axis=0)
        std = np.std(all_states, axis=0)
        return {'mean':m, 'std':std}
    else:
        return None


def load_walking(path, env, state, skip=1000, files='', mps=[0], norm_val=None, **kwargs):
    seconds2hours = 1 / 3600.0
    sm_batch_sz = 5 #30

    load = kwargs['load']

    angles, anglerates = split_state(state)
    if kwargs['use_state'] == 'all':
        pick = angles + anglerates
    elif kwargs['use_state'] == 'angles':
        pick = angles
    elif kwargs['use_state'] == 'anglerates':
        pick = anglerates

    #loding transitions
    if load == 0:
        dd = []
        min_len = np.inf
        for mp in mps:
            data = []
            timing = [np.array([0,0])]
            change = []
            for fname in files:
                fname_full = fname.format(path=path, env=env, mp=mp)
                if os.path.isfile(fname_full):
                    with open(fname_full, 'rb') as f:
                        d = pickle.load(f)
                        d[:, 0] = d[:, 0] + timing[-1][1]
                        timing.append( np.array([d[-sm_batch_sz*skip, 0], d[-1, 0]]) )
                        change.append(d.shape[0])
                        data.append(d)

            if len(data):
                data = np.concatenate(data)

                ts = data[::skip, 0] * seconds2hours
                timing = [t*seconds2hours for t in timing]
                change = [int(c/skip) for c in change[:-1]] # last element is the end of the whole trajectory
                # absorbing states are marked as 2
                fl = data[:, -1] - 1
                fl[fl<1] = 0
                cumfalls_all = np.cumsum(fl)
                cumfalls = cumfalls_all[::skip]
                states = extract_state(data, state, pick)[::skip, :]
                if norm_val:
                    #states = (states - norm_val['mean']) / norm_val['std']
                    states = states / norm_val['std']
                timesteps, sm = single(states, change, sm_batch_sz)
                timesteps_diff = timesteps.tolist()+[timesteps[-1]+1]
                fall_rate = (np.diff(cumfalls[timesteps_diff]) / np.diff(ts[timesteps_diff])) * seconds2hours
                ts = ts[timesteps]
                dd.append({ 'ts':ts, 'min':sm[:,0], 'mean':sm[:,1], 'max':sm[:,2], 'fl':fall_rate, 'cumfalls':cumfalls })
                min_len = min([min_len, len(timesteps)])

        # equalize sizes of arrays
        for d in dd:
            for key in d:
                if key not in ['cumfalls~~']:
                    if len(d[key]) > min_len:
                        d[key] = d[key][:min_len]

        with open('{path}/{env}-{fnum}.bin'.format(path=path, env=env, fnum=len(files)),'wb') as f:
            pickle.dump((dd, timing), f)
    else:
        with open('{path}/{env}-{fnum}.bin'.format(path=path, env=env, fnum=len(files)),'rb') as f:
            dd, timing = pickle.load(f)
    return dd, timing


def plot_separately(dd, ax, **kwargs):
    use_cols = kwargs["use_cols"] #['min', 'mean', 'max']
    xlog = kwargs["xlog"]
    ylog = kwargs["ylog"]
    hours2plot = kwargs["hours2plot"]

    mc = mean_confidence_dd(dd, use_cols)
    flmc =mean_confidence_dd(dd, ['fl'])

    ax.grid(True, color='lightgray', linestyle = '-', linewidth = 0.8)
    ax.set_axisbelow(True)

    #colors = cm.jet(np.linspace(0,1,len(use_cols)+1))
    if hours2plot:
        idx_to_plot = (mc['ts'] < hours2plot)
    else:
        idx_to_plot = range(len(mc['ts']))
    for i, col in enumerate(use_cols):
        ts = mc['ts'][idx_to_plot]
        mean = mc[col][0][idx_to_plot]
        ci = mc[col][1][idx_to_plot]
        ax.plot(ts, mean, label='surprise ('+col+")", color=palette[1])
        verts = list(zip(ts, mean+ci)) + list(zip(ts[::-1], (mean-ci)[::-1]))
        poly = Polygon(verts, facecolor=palette[1], alpha=0.3)
        ax.add_patch(poly)
    ax.set_ylabel('New states')
    ax.set_xlim(0, None)
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    ax2 = ax.twinx()

    ts = flmc['ts'][idx_to_plot]
    mean = flmc['fl'][0][idx_to_plot]
    ci = flmc['fl'][1][idx_to_plot]
    ax2.plot(ts, mean, color=palette[3], label='fall rate (mean)')
    verts = list(zip(ts, mean+ci)) + list(zip(ts[::-1], (mean-ci)[::-1]))
    poly = Polygon(verts, facecolor=palette[3], alpha=0.3)
    ax2.add_patch(poly)
    ax2.set_ylabel('Fall rate (fall/s)')
    if xlog:
        ax2.set_xscale('log')
    if ylog:
        ax2.set_yscale('log')

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    return ax, ax2


def plot_single(mc, ax, color, use_cols, idx_to_plot):
    for i, col in enumerate(use_cols):
        ts = mc['ts'][idx_to_plot]
        mean = mc[col][0][idx_to_plot]
        ci = mc[col][1][idx_to_plot]
        ax.plot(ts, mean, color=color)
        verts = list(zip(ts, mean+ci)) + list(zip(ts[::-1], (mean-ci)[::-1]))
        poly = Polygon(verts, facecolor=color, alpha=0.3)
        ax.add_patch(poly)

def surprise_simple(mc, flmc, name, **kwargs):
    surprise_threshold = kwargs["surprise_threshold"]
    hours2plot = kwargs["hours2plot"]
    stat_from = 0.2

    stat_interval = np.logical_and(mc['ts'] > stat_from, mc['ts'] < hours2plot)
    surprise = mc['mean'][0][stat_interval]
    high_surprise_idx = (surprise > surprise_threshold)

    fall_rate_infalls = np.diff(flmc['cumfalls'][0])
    fall_rate_infalls = np.concatenate((np.array([0]), fall_rate_infalls))[stat_interval]
    corresponding_fall_rate = fall_rate_infalls[high_surprise_idx]
    total_falls = np.sum(corresponding_fall_rate)
    print('###################')
    print("{}, surprise > {}, total falls = {}".format(name, surprise_threshold, total_falls))


def surprise_plot(dd, env, naming, colors, **kwargs):
    use_cols = kwargs["use_cols"]
    hours2plot = kwargs["hours2plot"]
    stat_from = 0.15

    surprise, fall_rate = [], []
    for d, color, name in zip(dd, colors, naming):
        mc = mean_confidence_dd(d, use_cols)
        flmc = mean_confidence_dd(d, ['cumfalls'])
        stat_interval = np.logical_and(mc['ts'] > stat_from, mc['ts'] < hours2plot)
        surprise.append(mc['mean'][0][stat_interval])

        fall_rate_infalls = np.diff(flmc['cumfalls'][0])
        fall_rate_infalls = np.concatenate((np.array([0]), fall_rate_infalls))
        fall_rate.append(fall_rate_infalls[stat_interval])

    surmin = min([min(surprise[0]), min(surprise[1])])
    surmax = max([max(surprise[0]), max(surprise[1])])
    surspace = np.linspace(surmin, surmax, 20)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    for sur, fl in zip(surprise, fall_rate):
        falls = []
        for surth in surspace:
            high_surprise_idx = (sur > surth)
            corresponding_fall_rate = fl[high_surprise_idx]
            falls.append(np.sum(corresponding_fall_rate))
        ax1.plot(surspace, falls)
    export_plot('tri_surprise_fall_rate_{}'.format(env), export_latex=False)



def plot_together(dd, timing, axarr, legends, naming, colors, **kwargs):
    use_cols = kwargs["use_cols"] #['min', 'mean', 'max']
    xlog = kwargs["xlog"]
    ylog = kwargs["ylog"]
    hours2plot = kwargs["hours2plot"]

    blowup = 0.8
    for d, color, name in zip(dd, colors, naming):
        mc = mean_confidence_dd(d, use_cols)
        flmc = mean_confidence_dd(d, ['fl', 'cumfalls'])

        if hours2plot:
            idx_to_plot = (mc['ts'] < hours2plot)
        else:
            idx_to_plot = range(len(mc['ts']))

        plot_single(mc, axarr[0], color, use_cols, idx_to_plot)
        plot_single(flmc, axarr[1], color, ['fl'], idx_to_plot)

        # blow-ups
        blowup_to_plot = (mc['ts'] < blowup)
        bx0 = plt.axes([.30, .8, .20, .15], facecolor='white')
        bx1 = plt.axes([.30, .35, .20, .15], facecolor='white')
        plot_single(mc, bx0, color, use_cols, blowup_to_plot)
        plot_single(flmc, bx1, color, ['fl'], blowup_to_plot)
        bx0.grid(True, color='lightgray', linestyle = '-', linewidth = 0.8)
        bx1.grid(True, color='lightgray', linestyle = '-', linewidth = 0.8)

        # calculate statistics
        surprise_simple(mc, flmc, name, **kwargs)

    labels = ('New states', 'Fall rate (fall/s)')
    for ax, label in zip(axarr, labels):
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        ax.grid(True, color='lightgray', linestyle = '-', linewidth = 0.8)
        ax.set_axisbelow(True)
        ax.set_ylabel(label)
        ax.get_yaxis().set_label_coords(-0.12, 0.5)
        end_time = mc['ts'][idx_to_plot][-1]
        ax.set_xlim((0, end_time))

    axarr[0].legend(legends, loc='upper right')


if __name__ == "__main__":
  main()
