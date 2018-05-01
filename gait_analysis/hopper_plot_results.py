"""RL data container."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib import cm
import matplotlib.collections as mcoll

sys.path.append('/home/ivan/work/scripts/py')
from my_csv.utils import get_header_size, listsubsample
from my_plot.plot import export_plot

plt.close('all')

subfigprop3 = {
  'figsize': (4.5, 3),
  'dpi': 80,
  'facecolor': 'w',
  'edgecolor': 'k'
}

offset = 1+8
state = OrderedDict([
        ('Time',                   0),
        ('HipAngle',        offset+0),
        ('HipAngleRate',    offset+1),
        ('KneeAngle',       offset+2),
        ('KneeAngleRate',   offset+3),
        ('AnkleAngle',      offset+4),
        ('AnkleAngleRate',  offset+5),

        ('Contact',         offset+6),

        ('HipControl',      offset+7),
        ('KneeControl',     offset+8),
        ('AnkleControl',    offset+9),
])

toplot = ['HipAngle', 'KneeAngle', 'AnkleAngle']
toplotlist = [state[key] for key in toplot]
tolegend = ['Hip', 'Knee', 'Ankle']

def main():
    folder = './'
    trajectories = folder + 'trajectories/'
    env = 'hopper'

    plot_results(env, trajectories, mp=5)

    #plot_histograms(env, trajectories, mp=2)

    #nearest_pose(env, trajectories, mp=2)

def split_state(modif=''):
    angles = []
    anglerates = []
    for key in state:
        if 'AngleRate' in key and modif in key:
            anglerates.append(key)
        elif 'Angle' in key and modif in key:
            angles.append(key)
    return (angles, anglerates)


def plot_histograms(env, path, mp):
    sd_walking = np.loadtxt(path + '{}_walking-mp{}.csv'.format(env, mp))
    sd_balancing = np.loadtxt(path + '{}_balancing-mp{}.csv'.format(env, mp))

    angles, anglerates = split_state()

    fig, axarr = plt.subplots(1, len(angles), **subfigprop3)
    for a, ax in zip(angles, axarr):
        sdw = sd_walking[:, state[a]]
        ax.hist(sdw)
        sdb = sd_balancing[:, state[a]]
        ax.hist(sdb)
        ax.set_xlabel(a)


def nearest_pose(env, path, mp):

    def get_nearest(angles, mode='nearest'):
        sd_walking = np.loadtxt(path + '{}_walking-mp{}.csv'.format(env, mp))
        sd_balancing = np.loadtxt(path + '{}_balancing-mp{}.csv'.format(env, mp))

        delay = 30
        sd_walking = sd_walking[delay:]
        sd_balancing = sd_balancing[delay:]

        sdw = []
        sdb = []
        for a in angles:
            sdw.append(sd_walking[:, state[a]])
            sdb.append(sd_balancing[-1, state[a]])
        sdw = np.column_stack(sdw)
        sdb = np.column_stack(sdb)

        dist = np.linalg.norm(sdw-sdb, axis=1)

        if mode == 'percentile':
            q = np.percentile(dist, 5)
            nearest = dist[dist<q]
            print('Nearest poses with values {}'.format(np.mean(nearest)))

        elif mode == 'nearest':
            nearest = np.argmin(dist)
            print('Nearest pose at time {} with value {}'.format(nearest, dist[nearest]))
            print(sdw[nearest,:])
            print(sdb)

    angles, _ = split_state()
    get_nearest(angles, mode='percentile')



def plot_results(env, path, mp):
    trajectory = np.loadtxt(path + '{}_walking-mp{}.csv'.format(env, mp))
    sd = trajectory[:, :state['AnkleAngleRate']+1]
    ud = trajectory[:, state['HipControl']:state['AnkleControl']+1]
    contacts = trajectory[:, state['Contact']]
    similarity = np.loadtxt(path + '{}_walking-mp{}_sim.csv'.format(env, mp))
    balancing_ud =  similarity[:, state['HipControl']:state['AnkleControl']+1]
    max_action = 1
    sim = np.abs(balancing_ud-ud) / (2*max_action)
    #sim /= np.max(sim)

    sim_mean = np.mean(sim)
    print('Similarity {}'.format(1-sim_mean))
    plot_trajectories(env, mp, sd, contacts, sim)


def plot_trajectories(env, mp, sd, contacts, sim):
    # shorten trajectory
    tr0 = -220
    tr1 = -50
    contacts = contacts[tr0:tr1]
    t0 = sd[tr0, state['Time']]
    sd = sd[tr0:tr1, :]
    sd[:, 0] = sd[:, state['Time']] - t0
    sim = sim[tr0:tr1, :]

    meansim = False
    if meansim:
        sim = np.mean(sim, axis=1)
    fig, axarr = plt.subplots(len(toplot), sharex=True, **subfigprop3)
    for name, idx, leg, ax in zip(toplot, toplotlist, tolegend, axarr):
        idx_sim = state[name.replace('Angle', 'Control')] - state['HipControl']
        print(idx)
        #idx_sim = idx - toplotlist[0]
        seg = [(sd[0, state['Time']], sd[0, idx])]
        for j in range(1, len(sd[:, state['Time']])):
            if (contacts[j] == 1):
                ax.axvspan(sd[j-1, state['Time']], sd[j, state['Time']], facecolor='0.8', alpha=1)
            seg.append((sd[j, state['Time']], sd[j, idx]))

        seg = np.array(seg)
        x, y = seg[:, 0], seg[:, 1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        #z = 1- sim[:, idx_sim] / 2.0
        if meansim:
            z = sim
        else:
            z = sim[:, idx_sim]
        lc = mcoll.LineCollection(segments, array=z, cmap='copper', linewidth=2) #, norm=plt.Normalize(0.0, 1.0)
        ax.add_collection(lc)
        cbaxes = fig.add_axes([0.87, 0.65, 0.02, 0.3])
        lc.set_clim(vmin=0, vmax=1)
        v = np.linspace(0, 1, 3, endpoint=True)
        plt.colorbar(lc, ax=ax, cax=cbaxes, ticks=v)
        ax.set_ylim(np.min(y), np.max(y))
        ax.get_yaxis().set_label_coords(-0.17, 0.5)
        ax.set_ylabel(leg)
        delta = 0.1*(max(sd[:, idx]) - min(sd[:, idx]))
        ax.set_ylim(min(sd[:, idx])-delta, max(sd[:, idx])+delta)
    axarr[-1].set_xlabel('Time (s)')
    plt.subplots_adjust(left=0.17, bottom=0.2, right=0.85, top=0.98, wspace=0.0, hspace=0.2)

    plt.show()
    export_plot('action_reuse_{}-mp{}'.format(env, mp))


if __name__ == "__main__":
  main()
