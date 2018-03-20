from __future__ import division
import numpy as np
import tensorflow as tf
import glob
import pickle
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from replaybuffer_ddpg import ReplayBuffer
from critic import CriticNetwork
from cl_network import CurriculumNetwork

tf.logging.set_verbosity(tf.logging.INFO)

from ptracker import PerformanceTracker
from cl_network import CurriculumNetwork
from ddpg import parse_args

tt  = 0 # duration
ee  = 1 # td error
cc  = 2 # complexity
ss  = 3 # curriculum stage
dim = 4

def read_file(f, cl_mode=0):
    try:
        data = np.loadtxt(f, skiprows=3, usecols=(3, 11, 12, 4, 5))
    except IndexError:
        return None
    except Exception as e:
        print(e)
    length = data.shape[0]
    damage = data[:,3,np.newaxis]
    distance = data[:,4,np.newaxis]
    data = np.hstack((data[:,0:3], cl_mode*np.ones((length,1))))
    return {'data':data, 'damage':damage, 'distance':distance}


def concat(first, second, first_no_damage=0):
    if not first and not second: return None
    if not first: return second
    if not second: return first
    data = np.vstack((first['data'], second['data']))
    damage1 = (1-first_no_damage)*first['damage']
    damage2 = damage1[-1] + second['damage']
    damage = np.vstack((damage1, damage2))
    distance = np.vstack((first['distance'], second['distance']))
    return {'data':data, 'damage':damage, 'distance':distance}


def load_data(path, params, gmax = 1):
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


def clean_data(dd, params):
    damamge_threshold = params['damamge_threshold']
    dd_new = []
    for d in dd:
        data = d['data']
        damage = d['damage']
        distance = d['distance']
        walked = True #any(distance > 20.0) # walked more then 10 m
        if walked and damamge_threshold and damage[-1] < damamge_threshold:
            if all(data[-1,1:] == data[-2,1:]): # last export was not due to the end of testing
                data = data[:-1, :]
                damage = damage[:-1, :]
            dd_new.append({'data': data, 'damage': damage, 'distance':distance})
    print('Percentage = {}'.format(len(dd_new)/len(dd)))
    return dd_new


def process_data(dd, config):
    dd_new = []
    for d in dd:
        data = d['data'] / np.array([config['env_timeout'], config["env_td_error_scale"], 1, 1])
        damage = d['damage'] #/ 1000.0 #/ config['default_damage']
        dd_new.append({'data':data, 'damage':damage, 'distance':d['distance']})
    return dd_new


def normalize(data, mean_data=None, std_data=None):
    if mean_data is None:
        mean_data = np.mean(data, axis=0)
    if std_data is None:
        std_data = np.std(data, axis=0)
    norm_data = (data-mean_data) / std_data
    return norm_data, mean_data, std_data


def normalize_data(dd, config, params, data_norm=None, damage_norm=None):
    steps_of_history = params['steps_of_history']
    zero_padding_after = params['zero_padding_after']
    dd_new = []

    # normalize
    if data_norm is None or damage_norm is None:
        data = []
        damage = []
        for d in dd:
            data_ = np.vstack((np.zeros((steps_of_history, dim)), d['data'], np.zeros((zero_padding_after, dim))))
            data_[0:steps_of_history, 3] = data_[steps_of_history, 3]
            data.append(data_)
            if params['damage_norm'] == 'to_reward':
                damage_ = -np.diff(d['damage'], axis=0)
                damage0 = -d['damage'][0]
                damage_ = np.vstack((np.ones((steps_of_history, 1))*damage0, damage_, np.zeros((1,1))))
            else:
                damage_ = d['damage'][-1] - d['damage']
                damage0 = np.array(d['damage'][-1])
                damage_ = np.vstack((np.ones((steps_of_history, 1))*damage0, damage_, np.zeros((1,1))))
            damage_ = np.vstack((damage_, np.zeros((zero_padding_after, 1))))
            damage.append(damage_)
        data, data_mean, data_std = normalize(np.concatenate(data))
        damage, damage_mean, damage_std = normalize(np.concatenate(damage))
        assert(len(np.where(~data.any(axis=1))[0]) == 0)    # assert no 0 rows in data, which is important for dynamic RNN
        #assert(len(np.where(~damage.any(axis=1))[0]) == 0)
    else:
        data_mean, data_std = data_norm
        damage_mean, damage_std = damage_norm

    # apply normalization to each episode
    for d in dd:
        data_ = np.vstack((np.zeros((steps_of_history, dim)), d['data'], np.zeros((zero_padding_after, dim))))
        data_[0:steps_of_history, 3] = data_[steps_of_history, 3]
        if params['indi_norm']:
            data_, _, _ = normalize(data_, data_mean, data_std)

        if params['damage_norm'] == 'to_reward':
            damage_ = -np.diff(d['damage'], axis=0)
            damage0 = -d['damage'][0]
            damage_ = np.vstack((np.ones((steps_of_history, 1))*damage0, damage_, np.zeros((1,1))))
        else:
            damage_ = (d['damage'][-1] - d['damage'])
            damage0 = d['damage'][-1]
            damage_ = np.vstack((np.ones((steps_of_history, 1))*damage0, damage_))
        damage_ = np.vstack((damage_, np.zeros((zero_padding_after, 1))))

        if 'norm' in params['damage_norm']:
            damage_, _, _ = normalize(damage_, damage_mean, damage_std)

        stage_ = d['data'][:, ss, np.newaxis]
        if params['stage_norm'] == 'cetered':
            stage_ = stage_ - 1
        stage_ = np.vstack((np.ones((steps_of_history, 1))*stage_[0], stage_, np.zeros((zero_padding_after, 1))))

        dd_new.append({'data': data_, 'damage': damage_, 'stage':stage_})

    return dd_new, (data_mean, data_std), (damage_mean, damage_std)


def seq_cut(dd, params):
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

def fill_replay_buffer(dd, config):
    replay_buffer = ReplayBuffer(config, o_dims=cc+1)
    for d in dd:
        data = d['data']
        damage = d['damage']
        stage = d['stage']
        for i in range(len(damage)-1):
            terminal = 1 if i == len(damage)-2 else 0
            replay_buffer.replay_buffer_add(data[i, tt:cc+1], stage[i], damage[i], terminal, data[i+1, tt:cc+1])
    assert(replay_buffer.replay_buffer_count < config["rb_max_size"])
    return replay_buffer


class cl(object):
    def __init__(self, size, config):
        self.network = CurriculumNetwork(size, config)
    def save(self, sess, name, global_step):
        self.network.save(sess, 'cl_network', global_step=global_step)
    def network_params(self):
        return self.network.network.network_params


class cl_critic(cl):
    def __init__(self, size, config):
        config['cl_structure'] = 'ffcritic:fc_relu_4;fc_relu_3;fc_relu_3'
        super().__init__(size, config)
    def predict(self, sess, s_batch, a_batch):
        return self.network.predict_(sess, s_batch, action=a_batch)
    def train(self, sess, s_batch, y_i, a_batch):
        self.network.train(sess, s_batch, y_i, action=a_batch)


class cl_critic_target(cl):
    def __init__(self, size, config):
        config["cl_target"] = True
        config['cl_structure'] = 'ffcritic:fc_relu_4;fc_relu_3;fc_relu_3'
        super().__init__(size, config)
    def predict(self, sess, s_batch, a_batch):
        return self.network.predict_target_(sess, s_batch, action=a_batch)
    def train(self, sess, s_batch, y_i, a_batch):
        self.network.train(sess, s_batch, y_i, action=a_batch)
        self.network.update_target_network(sess)

class cl_ff_regression(cl):
    def __init__(self, size, config):
        #config['cl_structure'] = 'ffr:fc_relu_;fc_linear_1'
        config['cl_structure'] = 'ffr:fc_linear_1'
        super().__init__(size+1, config) # action included to state
    def predict(self, sess, s_batch, a_batch):
        s_batch_new = np.concatenate((s_batch, a_batch), axis=1)
        return self.network.predict_(sess, s_batch_new)
    def train(self, sess, s_batch, y_i, a_batch):
        s_batch_new = np.concatenate((s_batch, a_batch), axis=1)
        self.network.train(sess, s_batch_new, y_i)


def main():
    params = {}
    params['steps_of_history'] = 1
    params['zero_padding_after'] = 0
    params['damamge_threshold'] = 10000.0
    params['indi_norm'] = True
    params['damage_norm'] = 'to_reward'
    params['stage_norm'] = 'cetered'
    params['neg_damage'] = True

    config = parse_args()
    config['default_damage'] = 4035.00
    config["cl_lr"] = 0.01
    config["cl_tau"] = 0.001
    #config['cl_structure'] = 'ffcritic:fc_relu_2;fc_relu_2;fc_relu_1'
#    config["cl_batch_norm"] = True
    config['cl_dropout_keep'] = 0.7
    config["cl_l2_reg"] = 0.000001
    config["minibatch_size"] = 128

    dd = load_data('leo_supervised_learning_regression/', params, gmax = 6)

    # Rule-based processing before splitting
    dd = clean_data(dd, params)
    dd = process_data(dd, config)

    # split into training and testing sets
    test_percentage = 0.3
    idx = np.arange(len(dd))
    np.random.shuffle(idx)
    test_idx = int(len(dd)*test_percentage)
    dd_train = [d for i,d in enumerate(dd) if i in idx[test_idx:]]
    dd_test = [d for i,d in enumerate(dd) if i in idx[:test_idx]]

    # normalize training dataset and use moments to normalizing test dataset
    dd_train, data_norm, damage_norm = normalize_data(dd_train, config, params)
    dd_test, _, _ = normalize_data(dd_test, config, params, data_norm, damage_norm)

    # save means and std for usage in RL
    pt = PerformanceTracker(depth=config['cl_depth'],
                            running_norm=config["cl_running_norm"],
                            input_norm=data_norm,
                            output_norm=damage_norm,
                            dim=3)
    pt.save('data_damage_norms.pkl')

    # get stat
    seq_cut(dd_train, params)
    seq_cut(dd_test, params)

    # fill in replay beuffer
    rb_train = fill_replay_buffer(dd_train, config)
    rb_test = fill_replay_buffer(dd_test, config)

    #config["minibatch_size"] = rb_train.replay_buffer_count

    with tf.Graph().as_default() as ddpg_graph:
        #cl_nn = cl_critic(pt.get_v_size(), config)
        #cl_nn = cl_critic_target(pt.get_v_size(), config)
        cl_nn = cl_ff_regression(pt.get_v_size(), config)


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    x, td_error_, mb_td_error_, train_td_error_, test_td_error_ = [], [], [], [], []
    plt.ion()
    with tf.Session(graph=ddpg_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # random initialization of variables
        sess.run(tf.global_variables_initializer())

        minibatch_size = config["minibatch_size"]
        for i in range(200000):
            s_batch, a_batch, r_batch, t_batch, s2_batch = rb_train.sample_batch(minibatch_size)

            # Calculate targets
            qq_val = []
            for stage in range(0,3):
                a_max = (stage-1)*np.ones((minibatch_size,1))
                qq_val.append(cl_nn.predict(sess, s2_batch, a_batch=a_max))
            q_val = np.concatenate(qq_val, axis=1)
            q_max = np.max(q_val, axis=1)
            q_max = np.reshape(q_max,newshape=(minibatch_size,1))

            y_i = []
            for k in range(minibatch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + config["gamma"] * q_max[k][0]) # target_q: list -> float

            if i%500 == 0:
                q_i = cl_nn.predict(sess, s_batch, a_batch=a_batch)
                td_error = np.sum(np.abs(q_i-np.reshape(y_i,newshape=(minibatch_size,1)))) / minibatch_size

            cl_nn.train(sess, s_batch, np.reshape(y_i, (minibatch_size,1)), a_batch=a_batch)

            # testing
            if i%500 == 0:
                not_biases = [ v for v in cl_nn.network_params() if '/b:' not in v.name ]
                print(sess.run(not_biases))

                print(min(q_max))

                mb_td_error = calc_td_error(sess, cl_nn, config, s_batch, a_batch, r_batch, t_batch, s2_batch, minibatch_size)

                s_batch, a_batch, r_batch, t_batch, s2_batch = rb_train.sample_batch(rb_train.replay_buffer_count)
                train_td_error = calc_td_error(sess, cl_nn, config, s_batch, a_batch, r_batch, t_batch, s2_batch, rb_train.replay_buffer_count)

                s_batch, a_batch, r_batch, t_batch, s2_batch = rb_test.sample_batch(rb_test.replay_buffer_count)
                test_td_error = calc_td_error(sess, cl_nn, config, s_batch, a_batch, r_batch, t_batch, s2_batch, rb_test.replay_buffer_count)

                print(td_error, mb_td_error, train_td_error, test_td_error)
                x.append(i)
                td_error_.append(td_error)
                mb_td_error_.append(mb_td_error)
                train_td_error_.append(train_td_error)
                test_td_error_.append(test_td_error)

                plt.plot(x, td_error_, 'r')
                plt.plot(x, mb_td_error_, 'g')
                plt.plot(x, train_td_error_, 'b')
                plt.plot(x, test_td_error_, 'k')
                plt.pause(0.05)

                if i%5000 == 0:
                    cl_nn.save(sess, 'cl_network', global_step=i)

    plt.show(block=True)



def calc_td_error(sess, cl_nn, config, s_batch, a_batch, r_batch, t_batch, s2_batch, size):
    # Calculate targets
    qq_val = []
    for stage in range(0,3):
        a_max = (stage-1)*np.ones((size,1))
        qq_val.append(cl_nn.predict(sess, s2_batch, a_batch=a_max))
    q_val = np.concatenate(qq_val, axis=1)
    q_max = np.max(q_val, axis=1)

    y_i = []
    for k in range(size):
        if t_batch[k]:
            y_i.append(r_batch[k])
        else:
            y_i.append(r_batch[k] + config["gamma"] * q_max[k]) # target_q: list -> float

    q_i = cl_nn.predict(sess, s_batch, a_batch=a_batch)
    td_error = np.sum(np.abs(q_i-np.reshape(y_i,newshape=(size,1))))
    td_error = td_error / size
    return td_error

######################################################################################
if __name__ == "__main__":
    main()

