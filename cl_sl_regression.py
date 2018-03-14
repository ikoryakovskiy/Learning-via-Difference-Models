from __future__ import division

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
clear_all()

import numpy as np
import tensorflow as tf
import glob
import pickle

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


def process_data(dd, config, steps_of_history):
    dd_new = []
    for d in dd:
        norm_duration = d['data'][:, 0, np.newaxis] / config['env_timeout']
        norm_duration = np.vstack((np.zeros((steps_of_history, 1)), norm_duration))
        norm_td_error = d['data'][:, 1, np.newaxis] / config["env_td_error_scale"]
        norm_td_error = np.vstack((np.zeros((steps_of_history, 1)), norm_td_error))
        norm_complexity = d['data'][:, 2, np.newaxis]
        norm_complexity = np.vstack((np.zeros((steps_of_history, 1)), norm_complexity))
        cl_mode = d['data'][:, 3, np.newaxis] - 1
        cl_mode = np.vstack((np.ones((steps_of_history, 1))*cl_mode[0], cl_mode))
        data = np.hstack((norm_duration, norm_td_error, norm_complexity, cl_mode))
        damage = (d['damage'][-1] - d['damage']) / config['default_damage']
        damage0 = d['damage'][-1] / config['default_damage']
        damage = np.vstack((np.ones((steps_of_history, 1))*damage0, damage))
        dd_new.append({'data': data, 'damage': damage})
    return dd_new

def seq_cut(dd, steps_of_history, dim):
    data_ = []
    damage_ = []
    for d in dd:
        data = d['data']
        damage = d['damage']
        for i in range(0, len(damage) - steps_of_history):
            data_.append(data[i:i+steps_of_history, :])
            #damage_ = damage[i:i+steps_of_history, :]
            damage_.append(damage[i+steps_of_history-1])
    data_ = np.reshape(data_, [-1, steps_of_history, dim])
    damage_ = np.reshape(damage_, [-1, 1])
    return {'data': data_, 'damage': damage_}

def main():
    steps_of_history = 5
    dim = 4

    config = parse_args()
    config["cl_lr"] = 0.001
    config['cl_structure'] = 'rnnr:rnn_tanh_3;fc_linear_1'
    config['cl_depth'] = 1
    config['default_damage'] = 4035.00

    prepare_datasets = False
    if prepare_datasets:
        dd = load_data('leo_supervised_learning_regression/')

        # Rule-based processing before splitting
        dd = process_data(dd, config, steps_of_history)

        # split into training and testing sets
        test_percentage = 0.3
        idx = np.arange(len(dd))
        np.random.shuffle(idx)
        test_idx = int(len(dd)*test_percentage)
        dd_test = [d for i,d in enumerate(dd) if i in idx[:test_idx]]
        dd_train = [d for i,d in enumerate(dd) if i in idx[test_idx:]]

        # cut data into sequences
        dd_test = seq_cut(dd_test, steps_of_history, dim)
        dd_train = seq_cut(dd_train, steps_of_history, dim)

        with open('data.pkl', 'wb') as f:
            pickle.dump((dd_train, dd_test), f)

    else:
        with open('data.pkl', 'rb') as f:
            dd_train, dd_test = pickle.load(f)

    # prepare for training
    pt = PerformanceTracker(depth=config['cl_depth'], input_norm=config["cl_input_norm"], dim=dim)
    cl_nn = CurriculumNetwork((steps_of_history, pt.get_v_size()), config)

    cl_nn.train(None, dd_train['data'], dd_train['damage'], n_epoch=150,
                validation_set=0.1, show_metric=True, batch_size=64)

    damage = cl_nn.predict_direct(None, dd_test['data'])
    error = np.linalg.norm(damage-dd_test['damage'])
    print(error)


######################################################################################
if __name__ == "__main__":
    main()

