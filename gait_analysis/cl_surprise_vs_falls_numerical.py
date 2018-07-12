"""RL data container."""

import os
import pickle
import colorsys
import numpy as np
from joblib import Parallel, delayed
from collections import OrderedDict

from pyrl.stat.stat import mean_confidence_dd, mean_confidence_interval

palette = ['#C0C0C0', '#4355bf', '#51843b', '#ffb171', '#408da9', '#408da9']

subfigprop3 = {
        'figsize': (4.2, 4.2),
        'dpi': 80,
        'facecolor': 'w',
        'edgecolor': 'k'
}

def hex_to_rgb(value, div = 1):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16)/div for i in range(0, lv, lv // 3))

def whiten(palette, alpha=0.3):
    hsv_palette = [colorsys.rgb_to_hsv(*hex_to_rgb(pp, div = 255)) for pp in palette]
    palette = [colorsys.hsv_to_rgb(pp[0], pp[1]*alpha, pp[2]) for pp in hsv_palette]
    #palette = [ [p/255 for p in pp] for pp in palette]
    return palette


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
    skip = 1000 #100
    kwargs = {
            "use_cols": ['mean'],
            "load": 0,
            "use_state":"all",
            "xlog": False,
            "ylog": False,
            "hours2plot": 2.0, #2
            "naming": ['direct', 'curriculum'],
            "mps": range(0,6)
            }

    leo_state, hopper_state, halfcheetah_state, walker2d_state = model_states()

    ###
    legends = ['direct', '\\pvrb']
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
    legends = ['direct', '\\pv']
    w =  ['{path}/ddpg-exp1_two_stage_{env}_ga_w-g0001-mp{mp}-02_walking.pkl']
    bw = [
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload-g0001-mp{mp}-01_balancing.pkl',
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload-g0001-mp{mp}-02_walking.pkl',
         ]
    #plot_single_system('hopper', w, bw, hopper_state, skip, legends, **kwargs)

    ###
    kwargs["surprise_threshold"] = 1.0
    w =  ['{path}/ddpg-exp1_two_stage_{env}_ga_w-g0001-mp{mp}-02_walking.pkl']
    bw = [
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload-g0001-mp{mp}-01_balancing.pkl',
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload-g0001-mp{mp}-02_walking.pkl',
         ]
    #plot_single_system('halfcheetah', w, bw, halfcheetah_state, skip, legends, **kwargs)

    ###
    kwargs["surprise_threshold"] = 1.0
    w =  ['{path}/ddpg-exp1_two_stage_{env}_ga_w-g0001-mp{mp}-02_walking.pkl']
    bw = [
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload-g0001-mp{mp}-01_balancing.pkl',
          '{path}/ddpg-exp1_two_stage_{env}_ga_bw-3_nnload-g0001-mp{mp}-02_walking.pkl',
         ]
    #plot_single_system('walker2d', w, bw, walker2d_state, skip, legends, **kwargs)


def plot_single_system(env, w, bw, state, skip, legends, mps, **kwargs):
    path = 'learning_trjectories'

    norm_val = min_max(path, env, state, skip=skip, ffiles=(w,bw), mps=mps, **kwargs)

    dd1, _ = load_walking(path, env, state, skip=skip, files=w, mps=mps, norm_val=norm_val, **kwargs)
    dd2, timing = load_walking(path, env, state, skip=skip, files=bw, mps=mps, norm_val=norm_val, **kwargs)

    hours2switch = timing[1][1]
    plot_together((dd1, dd2), hours2switch, legends, **kwargs)


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




def plot_together(dd, hours2switch, legends, naming, **kwargs):
    use_cols = kwargs["use_cols"][0] #['min', 'mean', 'max']
    hours2plot = kwargs["hours2plot"]

    cumsur = True
    #cumsur = False
    for d, name in zip(dd, naming):

        print(name)
        s1, s2, f1, f2 = [], [], [], []
        for rollout in d:
            stage1 = rollout['ts'] < hours2switch
            stage2 = (rollout['ts'] < hours2plot) & (rollout['ts'] >= hours2switch)

            sur1 = rollout[use_cols][stage1]
            sur2 = rollout[use_cols][stage2]
            if cumsur:
                sur1 = np.cumsum(sur1)[-1]
                sur2 = np.cumsum(sur2)[-1]
            else:
                sur1 = np.mean(sur1)
                sur2 = np.mean(sur2)

            fl1 = rollout['cumfalls'][stage1][-1]
            fl2 = rollout['cumfalls'][stage2][-1] - fl1

            s1.append(sur1)
            s2.append(sur2)
            f1.append(fl1)
            f2.append(fl2)

        print('  Balancing stage:')
        s = mean_confidence_interval(s1)
        f = mean_confidence_interval(f1)
        print('    At surprise {} cumulative number of falls is {}'.format(str(s), str(f)))

        print('  Walking stage:')
        s = mean_confidence_interval(s2)
        f = mean_confidence_interval(f2)
        print('    At surprise {} cumulative number of falls is {}'.format(str(s), str(f)))
    print('###################')


def find_idx_at(x, threshold = 0.63):
    sur_cumsum = np.cumsum(x)
    sur_th = sur_cumsum[-1]*threshold
    sur_last = np.nonzero(sur_cumsum < sur_th)[-1][-1]
    return sur_last, sur_cumsum[sur_last]


def calc_surprise(dd, hours2switch, legends, naming, **kwargs):
    use_cols = kwargs["use_cols"][0] #['min', 'mean', 'max']
    hours2plot = kwargs["hours2plot"]

    #cumsur = True
    cumsur = False
    joined_sur1, joined_sur2, joined_fl1, joined_fl2 = [], [], [], []
    named_indi_fl1, named_indi_fl2 = {}, {}
    for d, name in zip(dd, naming):

        indi_fl1, indi_fl2 = [], []
        for rollout in d:
            stage1 = rollout['ts'] < hours2switch
            stage2 = (rollout['ts'] < hours2plot) & (rollout['ts'] >= hours2switch)

            ts1 = rollout['ts'][stage1]
            ts2 = rollout['ts'][stage2] - rollout['ts'][stage2][0]

            sur1 = rollout[use_cols][stage1]
            sur2 = rollout[use_cols][stage2]
            if cumsur:
                sur1 = np.cumsum(sur1)
                sur2 = np.cumsum(sur2)
#            else:
#                sur1 = np.mean(sur1)
#                sur2 = np.mean(sur2)

            fl1 = rollout['cumfalls'][stage1]
            fl2 = rollout['cumfalls'][stage2] - rollout['cumfalls'][stage2][0]
            joined_sur1.append({'ts':ts1, 'sur1':sur1})
            joined_sur2.append({'ts':ts2, 'sur2':sur2})
            joined_fl1.append({'ts':ts1, 'fl1':fl1})
            joined_fl2.append({'ts':ts2, 'fl2':fl2})

            indi_fl1.append({'ts':ts1, 'fl1':fl1})
            indi_fl2.append({'ts':ts2, 'fl2':fl2})

        named_indi_fl1[name] = indi_fl1
        named_indi_fl2[name] = indi_fl2


    #find mean surprise during the first stage
    mc_sur1 = mean_confidence_dd(joined_sur1, ['sur1'], cutok=True)
    mc_sur2 = mean_confidence_dd(joined_sur2, ['sur2'], cutok=True)

    sur_idx1, mc_sur1_val = find_idx_at(mc_sur1['sur1'][0])
    sur_idx2, mc_sur2_val = find_idx_at(mc_sur2['sur2'][0])

    for name in naming:
        print('For {}:'.format(name))

        # find number of falls at given surprise
        mc_fl1 = mean_confidence_dd(named_indi_fl1[name], ['fl1'], cutok=True)
        mc_fl2 = mean_confidence_dd(named_indi_fl2[name], ['fl2'], cutok=True)

        print('  Balancing stage:')
        print('    At surprise {} cumulative number of falls is {} +/- {}'.format(
                mc_sur1_val,
                mc_fl1['fl1'][0][sur_idx1],
                mc_fl1['fl1'][1][sur_idx1],
                ))

        print('  Walking stage:')
        print('    At surprise {} cumulative number of falls is {} +/- {}'.format(
                mc_sur2_val,
                mc_fl2['fl2'][0][sur_idx2],
                mc_fl2['fl2'][1][sur_idx2],
                ))

    print('###################')




if __name__ == "__main__":
  main()
