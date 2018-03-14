from __future__ import division
import numpy as np
import tensorflow as tf
import glob

tf.logging.set_verbosity(tf.logging.INFO)

from ptracker import PerformanceTracker
from cl_network import CurriculumNetwork
from ddpg import parse_args


def read_file(f, cl_mode=0):
    try:
        data = np.loadtxt(f, skiprows=3, usecols=(3, 11, 12, 4))
    except IndexError:
        return None
    except Exception as e:
        print(e)
    length = data.shape[0]
    damage = data[:, -1,np.newaxis]
    data = np.hstack((data[:,0:3], cl_mode*np.ones((length,1))))
    return {'data':data, 'damage':damage}


def concat(first, second, first_no_damage=0):
    if not first and not second: return None
    if not first: return second
    if not second: return first
    data = np.vstack((first['data'], second['data']))
    damage1 = (1-first_no_damage)*first['damage']
    damage2 = damage1[-1] + second['damage']
    damage = np.vstack((damage1, damage2))
    return {'data': data, 'damage': damage}


def load_data(path, gmax = 1):
    stage_names = ('00_balancing_tf', '01_balancing', '02_walking')
    name_format = 'ddpg-g{:04d}-mp*-{stage_name}.monitor.csv'

    dd = []
    for g in range(1, 1+gmax):
        pat = path + name_format.format(g, stage_name=stage_names[0])
        for f in sorted(glob.glob(pat)):
            balancing_tf = read_file(f, cl_mode=0)
            balancing    = read_file(f.replace(stage_names[0], stage_names[1]), cl_mode=1)
            walking      = read_file(f.replace(stage_names[0], stage_names[2]), cl_mode=2)
            d = concat(balancing_tf, balancing)
            d = concat(d, walking)
            dd.append(d)
    return dd


def process_data(dd, config):
    dd_new = []
    for d in dd:
        norm_duration = d['data'][:, 0] / config['env_timeout']
        norm_td_error = d['data'][:, 1] / config["env_td_error_scale"]
        norm_complexity = d['data'][:, 2]
        cl_mode = d['data'][:, 3] - 1
        data = np.hstack((norm_duration[:,np.newaxis],
                          norm_td_error[:,np.newaxis],
                          norm_complexity[:,np.newaxis],
                          cl_mode[:,np.newaxis]))
        damage = (d['damage'][-1] - d['damage']) / config['default_damage']
        dd_new.append({'data': data, 'damage': damage})
    return dd_new


def main():
    config = parse_args()
    config["cl_lr"] = 0.0001
    config['cl_structure'] = 'rnnr:rnn_tanh_3;fc_linear_1'
    config['cl_depth'] = 1
    config['default_damage'] = 4035.00

    dd = load_data('leo_supervised_learning_regression/')

    # Rule-based processing before splitting
    dd = process_data(dd, config)

    # split into training and testing sets
    test_percentage = 0.3
    idx = np.arange(len(dd))
    np.random.shuffle(idx)
    test_idx = int(len(dd)*test_percentage)
    dd_test = [d for i,d in enumerate(dd) if i in idx[:test_idx]]
    dd_train = [d for i,d in enumerate(dd) if i in idx[test_idx:]]

    # prepare for training
    pt = PerformanceTracker(depth=config['cl_depth'], input_norm=config["cl_input_norm"], dim=4)
    cl_nn = CurriculumNetwork(pt.get_v_size(), config)

    cl_nn.train(dd_train['data'], dd_train['damage'])

    error = []
    for d in dd_train:
        damage = cl_nn.predict(dd_test['data'])
        error.append(np.linalg.norm(damage-dd_test['damage']))
    print( sum(error) / float(len(error)) )




######################################################################################
if __name__ == "__main__":
    main()

