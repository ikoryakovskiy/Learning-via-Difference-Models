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

plt.close('all')

subfigprop3 = {
  'figsize': (4.5, 3),
  'dpi': 80,
  'facecolor': 'w',
  'edgecolor': 'k'
}
"ffoot", "fshin", "fthigh",  "bfoot", "bshin", "bthigh"

offset = 1+8
state = OrderedDict([
        ('Time',                       0),
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

        ('FrontFootContact',    offset+12),
        ('FrontShinContact',    offset+13),
        ('FrontThighContact',   offset+14),
        ('BackFootContact',     offset+15),
        ('BackShinContact',     offset+16),
        ('BackThighContact',    offset+17),

        ('BackHipControl',      offset+18),
        ('BackKneeControl',     offset+19),
        ('BackAnkleControl',    offset+20),
        ('FrontHipControl',     offset+21),
        ('FrontKneeControl',    offset+22),
        ('FrontAnkleControl',   offset+23),
])

#toplot = ['BackHipAngle', 'BackKneeAngle', 'BackAnkleAngle', 'FrontHipAngle', 'FrontKneeAngle', 'FrontAnkleAngle']
#tolegend = ['BackHip', 'BackKnee', 'BackAnkle', 'FrontHip', 'FrontKnee', 'FrontAnkle']
toplot = ['FrontHipAngle', 'FrontKneeAngle', 'FrontAnkleAngle']
tolegend = ['Hip', 'Knee', 'Ankle']
toplotlist = [state[key] for key in toplot]

def main():
    folder = './'
    trajectories = folder + 'trajectories/'
    env = 'halfcheetah'

    plot_results(env, trajectories, mp=0)
    #plot_histograms(env, trajectories, mp=0)
    #nearest_pose(env, trajectories, mp=0)


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

    angles, _ = split_state('Left')
    get_nearest(angles, mode='percentile')
    angles, _ = split_state('Right')
    get_nearest(angles, mode='percentile')


def plot_results(env, path, mp):
    trajectory = np.loadtxt(path + '{}_walking-mp{}.csv'.format(env, mp))
    sd = trajectory[:, :state['FrontAnkleAngleRate']+1]
    ud = trajectory[:, state['BackHipControl']:state['FrontAnkleControl']+1]
    contacts = np.column_stack((trajectory[:, state['BackFootContact']], trajectory[:, state['FrontFootContact']]))
    similarity = np.loadtxt(path + '{}_walking-mp{}_sim.csv'.format(env, mp))
    balancing_ud =  similarity[:, state['BackHipControl']:state['FrontAnkleControl']+1]
    max_action = 1
    sim = np.abs(balancing_ud-ud) / (2*max_action)
    #sim /= np.max(sim)

    sim_mean = np.mean(sim)
    print('Similarity {}'.format(1-sim_mean))
    plot_trajectories(env, mp, sd, contacts, sim)


def plot_trajectories(env, mp, sd, contacts, sim):
    # shorten trajectory
    tr0 = -200
    tr1 = -100
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
        idx_sim = state[name.replace('Angle', 'Control')] - state['BackHipControl']
        print(idx)
        bf = 0 if 'Back' in name else 1
        seg = [(sd[0, state['Time']], sd[0, idx])]
        for j in range(1, len(sd[:, state['Time']])):
            if (contacts[j, bf] == 1):
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
            z = sim[:, idx_sim]
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidth=2) #, norm=plt.Normalize(0.0, 1.0)
        ax.add_collection(lc)
        cbaxes = fig.add_axes([0.82, 0.65, 0.02, 0.3])
        lc.set_clim(vmin=0, vmax=1)
        v = np.linspace(0, 1, 3, endpoint=True)
        plt.colorbar(lc, ax=ax, cax=cbaxes, ticks=v)
        ax.set_ylim(np.min(y), np.max(y))
        ax.get_yaxis().set_label_coords(-0.17, 0.5)
        ax.set_ylabel(leg)
        delta = 0.1*(max(sd[:, idx]) - min(sd[:, idx]))
        ax.set_ylim(min(sd[:, idx])-delta, max(sd[:, idx])+delta)
    axarr[-1].set_xlabel('Time (s)')
    plt.subplots_adjust(left=0.17, bottom=0.17, right=0.8, top=0.98, wspace=0.0, hspace=0.2)

    plt.show()
    export_plot('action_reuse_{}-mp{}'.format(env, mp))


if __name__ == "__main__":
  main()
