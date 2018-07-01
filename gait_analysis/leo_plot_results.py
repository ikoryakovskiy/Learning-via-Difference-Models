"""RL data container."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib import cm
import matplotlib.collections as mcoll

from matplotlib.colors import LinearSegmentedColormap

sys.path.append('/home/ivan/work/scripts/py')
from my_csv.utils import get_header_size, listsubsample
from my_plot.plot import export_plot

##############################################
import types

from matplotlib.backend_bases import GraphicsContextBase, RendererBase

class GC(GraphicsContextBase):
    def __init__(self):
        super().__init__()
        self._capstyle = 'round'

def custom_new_gc(self):
    return GC()

RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)
############################################

plt.close('all')

subfigprop3 = {
  'figsize': (4.5, 3),
  'dpi': 80,
  'facecolor': 'w',
  'edgecolor': 'k'
}

# These angles correspond to the actual colums in csv file.
# They are a bit different from definition in the task becasue they are exporting
# from python
state = OrderedDict([
        ('Time',             0),
        ('TorsoAngle',       1),
        ('LeftHipAngle',     2),
        ('RightHipAngle',    3),
        ('LeftKneeAngle',    4),
        ('RightKneeAngle',   5),
        ('LeftAnkleAngle',   6),
        ('RightAnkleAngle',  7)
])

#toplot = ['TorsoZ', 'TorsoAngle', 'LeftHipAngle', 'LeftKneeAngle', 'LeftAnkleAngle']
toplot = ['LeftHipAngle', 'LeftKneeAngle', 'LeftAnkleAngle']
#toplot = ['LeftHipAngle', 'LeftKneeAngle', 'LeftAnkleAngle', 'RightHipAngle', 'RightKneeAngle', 'RightAnkleAngle']
toplotlist = [state[key] for key in toplot]
tolegend = ['Hip', 'Knee', 'Ankle']

def main():
    folder = './'
    trajectories = folder + 'trajectories/'

    plot_results(trajectories, mp=0)

    #plot_histograms(trajectories, mp=2)

    #nearest_pose(trajectories, mp=1)

def split_state(modif=''):
    angles = []
    anglerates = []
    for key in state:
        if 'AngleRate' in key and modif in key:
            anglerates.append(key)
        elif 'Angle' in key and modif in key:
            angles.append(key)
    return (angles, anglerates)


def plot_histograms(path, mp):
    sd_walking = np.loadtxt(path + 'leo_walking-mp{}.csv'.format(mp))
    sd_balancing = np.loadtxt(path + 'leo_balancing-mp{}.csv'.format(mp))

    angles, anglerates = split_state()

    fig, axarr = plt.subplots(1, len(angles), **subfigprop3)
    for a, ax in zip(angles, axarr):
        sdw = sd_walking[:, state[a]]
        ax.hist(sdw)
        sdb = sd_balancing[:, state[a]]
        ax.hist(sdb)
        ax.set_xlabel(a)


def nearest_pose(path, mp):

    def get_nearest(angles, mode='nearest'):
        sd_walking = np.loadtxt(path + 'leo_walking-mp{}.csv'.format(mp))
        sd_balancing = np.loadtxt(path + 'leo_balancing-mp{}.csv'.format(mp))

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

    angles, _ = split_state('Left')
    get_nearest(angles, mode='percentile')
    angles, _ = split_state('Right')
    get_nearest(angles, mode='percentile')




def plot_results(path, mp):

    contacts = np.genfromtxt(path + 'aux_leo-mp{}.csv'.format(mp), delimiter=',', skip_header=1)
    contacts = contacts[:-1, 1]
    trajectory = np.loadtxt(path + 'leo_walking-mp{}.csv'.format(mp))
    sdim = 2*7
    sd = trajectory[:, :1+sdim]
    ud = trajectory[:, 1+sdim:1+sdim+6]
    similarity = np.loadtxt(path + 'leo_walking-mp{}_sim.csv'.format(mp))
    balancing_ud =  similarity[:, 1+sdim:1+sdim+6]
    max_action = 10.69
    sim = np.abs(balancing_ud-ud) / (2*max_action)
    #sim /= np.max(sim)

    # shorten trajectory
    tr0 = -100
    tr1 = -10
    contacts = contacts[tr0:tr1]
    t0 = sd[tr0, state['Time']]
    sd = sd[tr0:tr1, :]
    sd[:, 0] = sd[:, state['Time']] - t0
    sim = sim[tr0:tr1, :]

    cmap = LinearSegmentedColormap.from_list("", ["black","#80d5ff"])


    meansim = False
    if meansim:
        sim = np.mean(sim, axis=1)
    fig, axarr = plt.subplots(len(toplot), sharex=True, **subfigprop3)
    for name, idx, leg, ax in zip(toplot, toplotlist, tolegend, axarr):
        idx_ud = idx - toplotlist[0]
        seg = [(sd[0, state['Time']], sd[0, idx])]
        for j in range(1, len(sd[:, state['Time']])):
            if 'Left' in name:
                if (contacts[j] == 1000) or (contacts[j] == 100) or (contacts[j] == 1100):
                    ax.axvspan(sd[j-1, state['Time']], sd[j, state['Time']], facecolor='0.9', alpha=1)
            else:
                if (contacts[j] == 1) or (contacts[j] == 10) or (contacts[j] == 11):
                    ax.axvspan(sd[j-1, state['Time']], sd[j, state['Time']], facecolor='0.9', alpha=1)

            seg.append((sd[j, state['Time']], sd[j, idx]))

        seg = np.array(seg)
        x, y = seg[:, 0], seg[:, 1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        #z = 1- sim[:, idx_sim] / 2.0
        if meansim:
            z = sim
        else:
            z = sim[:, idx_ud]
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidth=2) #, norm=plt.Normalize(0.0, 1.0)
        ax.add_collection(lc)
        cbaxes = fig.add_axes([0.87, 0.65, 0.02, 0.3])
        lc.set_clim(vmin=0, vmax=1)
        v = np.linspace(0, 1, 3, endpoint=True)
        plt.colorbar(lc, ax=ax, cax=cbaxes, ticks=v)
        ax.set_ylim(np.min(y), np.max(y))
        ax.get_yaxis().set_label_coords(-0.12, 0.5)
        ax.set_ylabel(leg)
    axarr[-1].set_xlabel('Time (s)')
    plt.subplots_adjust(left=0.17, bottom=0.2, right=0.85, top=0.98, wspace=0.0, hspace=0.2)

    plt.show()
    export_plot('action_reuse_leo-mp{}'.format(mp))


if __name__ == "__main__":
  main()
