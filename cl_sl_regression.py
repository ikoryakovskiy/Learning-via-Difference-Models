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
from sklearn.metrics import r2_score

tf.logging.set_verbosity(tf.logging.INFO)

from ptracker import PerformanceTracker
from cl_network import CurriculumNetwork
from ddpg import parse_args

tt = 0 # duration
ee = 1 # td error
cc = 2 # complexity
ss = 3 # curriculum stage


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


def clean_dataset(dd, params):
    damamge_threshold = params['damamge_threshold']
    dd_new = []
    for d in dd:
        if damamge_threshold and d['damage'][-1] < damamge_threshold:
            dd_new.append(d)
    return dd_new


def process_data(dd, config, params, zero_padding_after=1):
    steps_of_history = params['steps_of_history']
    dd_new = []
    for d in dd:
        norm_duration = d['data'][:, 0, np.newaxis] / config['env_timeout']
        norm_duration = np.vstack((np.zeros((steps_of_history, 1)), norm_duration))
        norm_duration = np.vstack((norm_duration, np.zeros((zero_padding_after, 1))))

        norm_td_error = d['data'][:, 1, np.newaxis] / config["env_td_error_scale"]
        norm_td_error = np.vstack((np.zeros((steps_of_history, 1)), norm_td_error))
        norm_td_error = np.vstack((norm_td_error, np.zeros((zero_padding_after, 1))))

        norm_complexity = d['data'][:, 2, np.newaxis]
        norm_complexity = np.vstack((np.zeros((steps_of_history, 1)), norm_complexity))
        norm_complexity = np.vstack((norm_complexity, np.zeros((zero_padding_after, 1))))

        cl_mode = (d['data'][:, 3, np.newaxis] + 1)/10
        cl_mode = np.vstack((np.ones((steps_of_history, 1))*cl_mode[0], cl_mode))
        cl_mode = np.vstack((cl_mode, np.zeros((zero_padding_after, 1))))

        data = np.hstack((norm_duration, norm_td_error, norm_complexity, cl_mode))

        damage = (d['damage'][-1] - d['damage']) / config['default_damage']
        damage0 = d['damage'][-1] / config['default_damage']
        damage = np.vstack((np.ones((steps_of_history, 1))*damage0, damage))
        damage = np.vstack((damage, np.zeros((zero_padding_after, 1)))) # No padding of damage?

        dd_new.append({'data': data, 'damage': damage})

    return dd_new

def normalize(data, mean_data=None, std_data=None):
    if mean_data is None:
        mean_data = np.mean(data, axis=0)
    if std_data is None:
        std_data = np.std(data, axis=0)
    norm_data = (data-mean_data[np.newaxis,:]) / std_data[np.newaxis,:]
    return norm_data, mean_data, std_data

def normalize_data(dd, config, params, zero_padding_after=1, data_norm=None, damage_norm=None):
    steps_of_history = params['steps_of_history']
    dd_new = []

    # normalize
    if data_norm is None or damage_norm is None:
        data = []
        damage = []
        for d in dd:
            data.append(d['data'])
            damage_ = d['damage'][-1] - d['damage']
            damage0 = np.array(d['damage'][-1])
            damage.append(np.vstack((damage0, damage_)))
        data, data_mean, data_std = normalize(np.concatenate(data))
        damage, damage_mean, damage_std = normalize(np.concatenate(damage))
        assert(len(np.where(~data.any(axis=1))[0]) == 0)    # assert no 0 rows in data, which is important for dynamic RNN
        assert(len(np.where(~damage.any(axis=1))[0]) == 0)
    else:
        data_mean, data_std = data_norm
        damage_mean, damage_std = damage_norm

    # apply normalization to each episode
    dim = len(data_mean)
    for d in dd:
        data, _, _ = normalize(d['data'], data_mean, data_std)
        data = np.vstack((np.zeros((steps_of_history, dim)), data, np.zeros((zero_padding_after, dim))))
        data[0:steps_of_history, 3] = data[steps_of_history, 3]

        damage = (d['damage'][-1] - d['damage'])
        damage0 = d['damage'][-1]
        damage = np.vstack((np.ones((steps_of_history, 1))*damage0, damage))
        damage, _, _ = normalize(damage, damage_mean, damage_std)
        damage = np.vstack((damage, np.zeros((zero_padding_after, 1)))) # No padding of damage?

        stage = d['data'][:, ss, np.newaxis]
        stage = np.vstack((np.ones((steps_of_history, 1))*stage[0], stage, -1*np.ones((zero_padding_after, 1))))

        dd_new.append({'data': data, 'damage': damage, 'stage':stage})

    return dd_new, (data_mean, data_std), (damage_mean, damage_std)


def seq_cut(dd, params, dim):
    steps_of_history = params['steps_of_history']
    data_ = []
    damage_ = []
    stage_ = []
    seq_data_ = []
    seq_damage_ = []
    for d in dd:
        data = d['data']
        damage = d['damage']
        stage = d['stage']
        for i in range(0, len(damage) - steps_of_history + 1):
            seq_data_.append(data[i:i+steps_of_history, :])
            seq_damage_.append(damage[i+steps_of_history-1]) #damage_ = damage[i:i+steps_of_history, :]
            data_.append(data[i+steps_of_history-1, :])
            damage_.append(damage[i+steps_of_history-1])
            stage_.append(stage[i+steps_of_history-1])

    seq_data_ = np.reshape(seq_data_, [-1, steps_of_history, dim])
    seq_damage_ = np.reshape(seq_damage_, [-1, 1])
    data_ = np.array(data_)
    damage_ = np.array(damage_)
    stage_ = np.array(stage_)

    min_data_ = np.min(data_, axis=0)
    med_data_ = np.median(data_, axis=0)
    max_data_ = np.max(data_, axis=0)
    min_damage_ = np.min(damage_, axis=0)
    med_damage_ = np.median(damage_, axis=0)
    max_damage_ = np.max(damage_, axis=0)
    print('Data stat:\n  {}\n  {}\n  {}\n  {}\n  {}\n  {}\n '.format(min_data_, med_data_, max_data_,
          min_damage_, med_damage_, max_damage_))
    return {'seq_data': seq_data_, 'seq_damage': seq_damage_, 'data': data_, 'damage': damage_, 'stage': stage_}

def main():
    params = {}
    params['steps_of_history'] = 3
    params['damamge_threshold'] = 10000.0
    params['normalize'] = True
    zero_padding_after = 1
    dim = 4

    config = parse_args()
    config["cl_lr"] = 0.001
    config['cl_structure'] = 'rnnr:rnn_tanh_8_dropout;fc_linear_1'
#    config['cl_structure'] = 'rnnr:rnn_tanh_4_dropout;rnn_tanh_4_dropout;fc_linear_1'
    config['cl_dropout_keep'] = 0.7
    config['cl_depth'] = 1
    config['default_damage'] = 4035.00


    prepare_datasets = True
    if prepare_datasets:
        dd = load_data('leo_supervised_learning_regression/')

        # Rule-based processing before splitting
        dd = clean_dataset(dd, params)
        #dd = process_data(dd, config, params, zero_padding_after)

        # split into training and testing sets
        test_percentage = 0.3
        idx = np.arange(len(dd))
        np.random.shuffle(idx)
        test_idx = int(len(dd)*test_percentage)
        dd_train = [d for i,d in enumerate(dd) if i in idx[test_idx:]]
        dd_test = [d for i,d in enumerate(dd) if i in idx[:test_idx]]

        # normalize training dataset and use moments to normalizing test dataset
        dd_train, data_norm, damage_norm = normalize_data(dd_train, config, params, zero_padding_after)
        dd_test, _, _ = normalize_data(dd_test, config, params, zero_padding_after, data_norm, damage_norm)

        # cut data into sequences
        dd_train = seq_cut(dd_train, params, dim)
        dd_test = seq_cut(dd_test, params, dim)

        with open('data.pkl', 'wb') as f:
            pickle.dump((dd_train, dd_test), f)

    else:
        with open('data.pkl', 'rb') as f:
            dd_train, dd_test = pickle.load(f)

    # prepare for training
    pt = PerformanceTracker(depth=config['cl_depth'], input_norm=config["cl_running_norm"], dim=dim)
    cl_nn = CurriculumNetwork((params['steps_of_history'], pt.get_v_size()), config)

    cl_nn.train(None, dd_train['seq_data'], dd_train['seq_damage'], n_epoch=2,
                validation_set=0.1, show_metric=True, batch_size=64)


    ### ACCURACY
    data = dd_test['data']
    seq_end_idx = np.where(~data.any(axis=1))[0]
    true_stage = dd_test['stage']
    true_stage = np.asarray(np.rint(true_stage), dtype=int)
    true_stage[seq_end_idx] = 0
    true_stage_damage = dd_test['damage']

    damage = []
    for i in range(3):
        seq_data = dd_test['seq_data']
        seq_data[:, :, -1] = (i+1)/10
        seq_data[seq_end_idx, -1, -1] = 0
        damage.append(cl_nn.predict_(None, seq_data))
        #verify = cl_nn.predict_(None, seq_data)
    damage_ = np.transpose(np.reshape(damage, [3, -1]))
    cl_predict = np.argmin(damage_, axis=1)
    #cl_predict_damage_min = np.min(damage_, axis=1)
    cl_predict_damage = damage_[range(len(true_stage)), true_stage]
    seq_end_idx_ = np.concatenate((-1*np.ones((1,)), seq_end_idx))
    seq_end_idx_ = np.asarray(seq_end_idx_, dtype=int)
    seq_end_idx_range = range(len(seq_end_idx_)-1)
    predict_damage = np.array([sum(cl_predict_damage[seq_end_idx_[i]+1:seq_end_idx_[i+1]]) for i in seq_end_idx_range])
    true_damage = np.array([sum(true_stage_damage[seq_end_idx_[i]+1:seq_end_idx_[i+1]]) for i in seq_end_idx_range])
    true_damage_sorted_idx = np.argsort(true_damage, axis=0)
    predict_damage_sorted = predict_damage[true_damage_sorted_idx]
    print(predict_damage_sorted)

    error = np.linalg.norm(predict_damage-true_damage)
    coefficient_of_dermination = r2_score(true_damage, predict_damage)
    print(error, coefficient_of_dermination)


#    true_stage = data[:, -1]*10 - 1
#    true_stage = np.asarray(np.rint(true_stage), dtype=int)
#    diff = true_stage-cl_predict
#    diff[seq_end_idx] = 0
#    accuracy = 1 - np.count_nonzero(diff)/(len(true_stage)-len(seq_end_idx))
#    print("Accuracy {}".format(accuracy))


    ### ERROR and R-SQUARED
    damage = cl_nn.predict_(None, dd_test['seq_data'])
    error = np.linalg.norm(damage-dd_test['seq_damage'])
    coefficient_of_dermination = r2_score(dd_test['seq_damage'], damage)
    print(error, coefficient_of_dermination)


#    # PREDICT SINGLE
#    stage, rr = [], []
#    for d in dd_test['data']:
#        d = np.reshape(d, [-1, steps_of_history, dim])
#        stage_, rr_ = cl_nn.predict(None, d)
#        stage.append(stage_)
#        rr.append(rr_)


######################################################################################
if __name__ == "__main__":
    main()

