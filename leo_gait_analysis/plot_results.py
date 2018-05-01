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
  'figsize': (5, 3),
  'dpi': 80,
  'facecolor': 'w',
  'edgecolor': 'k'
}

state = OrderedDict([
        ('Time',             0),
        ('TorsoX',           1),
        ('TorsoZ',           2),
        ('TorsoAngle',       3),
        ('LeftHipAngle',     4),
        ('RightHipAngle',    5),
        ('LeftKneeAngle',    6),
        ('RightKneeAngle',   7),
        ('LeftAnkleAngle',   8),
        ('RightAnkleAngle',  9)
])

#toplot = ['TorsoZ', 'TorsoAngle', 'LeftHipAngle', 'LeftKneeAngle', 'LeftAnkleAngle']
toplot = ['LeftHipAngle', 'LeftKneeAngle', 'LeftAnkleAngle']
toplotlist = [state[key] for key in toplot]
tolegend = ['Hip', 'Knee', 'Ankle']

def main():
    plot_results('wb')
    #plot_results('cb')

def plot_results(path):

    path = path + '/'
    sd = np.loadtxt(path + 'sd_leo.csv', delimiter=',')
    sd = sd[:-1, :]
    wb = np.loadtxt(path + 'sim_leo_wb.csv')

    # shorten trajectory
    trlen = 200
    sd = sd[-trlen:,:]
    wb = wb[-trlen:,:]

    with open(path + 'aux_leo.csv') as f:
        phase = f.readlines()

    num = len(toplotlist)
    fig, axarr = plt.subplots(num, sharex=True, **subfigprop3)
    for i in range(num):
        actuator_idx = toplotlist[i] - state['LeftHipAngle']
        if actuator_idx < 0:
            axarr[i].plot(sd[:, state['Time']], sd[:, toplotlist[i]])
        else:
            seg = [(sd[0, state['Time']], sd[0, toplotlist[i]])]
            for j in range(1, len(sd[:, state['Time']])):

                if ('1000' in phase[j]) or ('0100' in phase[j]):
                    axarr[i].axvspan(sd[j-1, state['Time']], sd[j, state['Time']], facecolor='0.8', alpha=1)

                if '1100' in phase[j]:
                    axarr[i].axvspan(sd[j-1, state['Time']], sd[j, state['Time']], facecolor='0.8', alpha=1)

                seg.append((sd[j, state['Time']], sd[j, toplotlist[i]]))
            seg = np.array(seg)
            x, y = seg[:, 0], seg[:, 1]
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            z = 1- wb[:, 1+actuator_idx] / 2.0
            lc = mcoll.LineCollection(segments, array=z, cmap='copper', norm=plt.Normalize(0.0, 1.0))
            axarr[i].add_collection(lc)
            cbaxes = fig.add_axes([0.87, 0.45, 0.02, 0.5])
            plt.colorbar(lc, ax=axarr[i], cax=cbaxes)
            axarr[i].set_ylim(np.min(y), np.max(y))
            axarr[i].get_yaxis().set_label_coords(-0.14, 0.5)
        axarr[i].set_ylabel(tolegend[i])
    axarr[-1].set_xlabel('Time (s)')
    plt.subplots_adjust(left=0.14, bottom=0.2, right=0.85, top=0.98, wspace=0.0, hspace=0.2)

    plt.show()
    export_plot('leo_action_reuse')


if __name__ == "__main__":
  main()
