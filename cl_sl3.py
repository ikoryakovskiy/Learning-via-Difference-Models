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
#ss  = 3 # curriculum stage
dim = 3
ns = 3 # number of curriculum stages

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
    data = data[:,0:3]
    stage = cl_mode*np.ones((length,1))

    # last export was not due to the end of testing
    if all(data[-1,1:] == data[-2,1:]):
        data = data[:-1, :]
        damage = damage[:-1, :]
        distance = distance[:-1, :]

    return {'data':data, 'stage':stage, 'damage':damage, 'distance':distance}


def concat(first, second, first_no_damage=0):
    if not first and not second: return None
    if not first: return second
    if not second: return first
    data = np.vstack((first['data'], second['data']))
    stage = np.vstack((first['stage'], second['stage']))
    damage1 = (1-first_no_damage)*first['damage']
    damage2 = damage1[-1] + second['damage']
    damage = np.vstack((damage1, damage2))
    distance = np.vstack((first['distance'], second['distance']))
    return {'data':data, 'stage':stage, 'damage':damage, 'distance':distance}


def load_data(path, params, gens = [1]):
    stage_names = ('00_balancing_tf', '01_balancing', '02_walking')
    name_format = 'ddpg-g{:04d}-mp*-{stage_name}.monitor.csv'

    dd = []
    for g in gens:
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
        stage = d['stage']
        damage = d['damage']
        distance = d['distance']
        walked = True #any(distance > 20.0) # walked more then 10 m
        if walked and damamge_threshold and damage[-1] < damamge_threshold:
            dd_new.append({'data':data, 'stage':stage, 'damage':damage})
    print('Percentage = {}'.format(len(dd_new)/len(dd)))
    return dd_new


def process_data(dd, config):
    dd_new = []
    for d in dd:
        data = d['data'] / np.array([config['env_timeout'], config["env_td_error_scale"], 1])
        stage = d['stage']
        damage = d['damage'] #/ 1000.0 #/ config['default_damage']
        dd_new.append({'data':data, 'stage':stage, 'damage':damage})
    return dd_new


def select_good(dd, config, good_ratio):
    final_damage = []
    for d in dd:
        final_damage.append(d['damage'][-1,0])
    idx = np.argsort(final_damage)
    threshold = int(good_ratio*len(idx))
    good_dd_idx = idx[:threshold]
    good_dd = [d for i, d in enumerate(dd) if i in good_dd_idx]
    return good_dd


def normalize(data, mean_data=None, std_data=None):
    if mean_data is None:
        mean_data = np.mean(data, axis=0)
    if std_data is None:
        std_data = np.std(data, axis=0)
    norm_data = (data-mean_data) / std_data
    return norm_data, mean_data, std_data


def missing_data(data):
    '''Checks if some indicators in the beginning are missing due to wait for replay buffer filling'''
    hlen = data.shape[0]
    begin = 0
    for i in range(hlen):
        if all(data[i, 1:3] == 0):
            begin = i+1
        else:
            break
    return begin


def normalize_data(dd, config, params, data_norm=None, damage_norm=None):
    steps_of_history = params['steps_of_history']
    zero_padding_after = params['zero_padding_after']
    dd_new = []

    # normalize
    if data_norm is None or damage_norm is None:
        data = []
        damage = []
        for d in dd:
            begin = missing_data(d['data'])
            data_ = d['data'][begin:, :]
            data_ = np.vstack((np.zeros((steps_of_history, dim)), data_))
            data_ = np.vstack((data_, np.zeros((zero_padding_after, dim))))
            data.append(data_)

            damage_ = d['damage'][begin:, :]
            if params['damage_norm'] == 'to_reward':
                damage_ = -np.diff(damage_, axis=0)
                damage0 = -damage_[0]
                damage_ = np.vstack((np.ones((steps_of_history, 1))*damage0, damage_, np.zeros((1,1))))
            else:
                damage_ = damage_[-1] - damage_
                damage0 = np.array(damage_[-1])
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
        begin = missing_data(d['data'])
        data_ = d['data'][begin:, :]
        data_ = np.vstack((np.zeros((steps_of_history, dim)), data_))
        if params['indi_norm']:
            data_, _, _ = normalize(data_, data_mean, data_std)
        data_ = np.vstack((data_, np.zeros((zero_padding_after, dim))))

        damage_ = d['damage'][begin:, :]
        if params['damage_norm'] == 'to_reward':
            damage_ = -np.diff(damage_, axis=0)
            damage0 = -damage_[0]
            damage_ = np.vstack((np.ones((steps_of_history, 1))*damage0, damage_, np.zeros((1,1))))
            #damage_1 = np.exp(-np.diff(damage_, axis=0)/30.0)
            #damage_2 = np.exp(-damage_[0]/30.0)
            #damage_ = np.vstack((np.ones((steps_of_history, 1))*damage_2, damage_1, np.zeros((1,1))))
        else:
            damage_ = (damage_[-1] - damage_)
            damage0 = damage_[-1]
            damage_ = np.vstack((np.ones((steps_of_history, 1))*damage0, damage_))
        damage_ = np.vstack((damage_, np.zeros((zero_padding_after, 1))))

        if 'norm' in params['damage_norm']:
            damage_, _, _ = normalize(damage_, damage_mean, damage_std)

        stage_ = d['stage'][begin:]
        if params['stage_norm'] == 'cetered':
            stage_ = stage_ - 1
        stage_ = np.vstack((np.ones((steps_of_history, 1))*stage_[0], stage_, np.zeros((zero_padding_after, 1))))

        # softmax labels (shifted by 1 to predict new switch of the curriculum)
        classes = 3
        labels_ = np.concatenate((d['stage'][begin+1:], d['stage'][-1,np.newaxis]))[:,0] # <-- shifting
        labels_ = np.asarray(labels_, dtype=int)
        softmax_stage_ = np.zeros((len(labels_), classes))
        softmax_stage_[np.arange(len(labels_)), labels_] = 1
        history_block = np.array([softmax_stage_[0,:],]*steps_of_history)
        softmax_stage_ = np.vstack((history_block, softmax_stage_, np.zeros((zero_padding_after, classes))))

        dd_new.append({'data': data_, 'damage': damage_, 'stage':stage_, 'softmax_stage':softmax_stage_})

    return dd_new, (data_mean, data_std), (damage_mean, damage_std)


def calc_class_weight(dd, params):
    steps_of_history = params['steps_of_history']
    zero_padding_after = params['zero_padding_after']
    class_counter = np.zeros([ns,1])
    for d in dd:
        for ss in d['stage'][steps_of_history:-zero_padding_after]:
            class_counter[int(ss)] += 1
    return (sum(class_counter) - class_counter) / sum(class_counter)

def seq_cut(dd, params):
    if len(dd)==0:
        return []
    steps_of_history = params['steps_of_history']
    data_ = []
    damage_ = []
    stage_ = []
    dd_new = []
    for d in dd:
        data = d['data']
        damage = d['damage']
        stage = d['stage']
        softmax_stage = d['softmax_stage']
        seq_data_ = []
        seq_damage_ = []
        seq_softmax_stage_ = []
        for i in range(0, len(damage) - steps_of_history + 1):
            seq_data_.append(data[i:i+steps_of_history, :])
            seq_damage_.append(damage[i+steps_of_history-1])
            seq_softmax_stage_.append(softmax_stage[i+steps_of_history-1, :])
            data_.append(data[i+steps_of_history-1, :])
            damage_.append(damage[i+steps_of_history-1])
            stage_.append(stage[i+steps_of_history-1])
        dd_new.append({'seq_data': seq_data_, 'seq_damage': seq_damage_, 'seq_softmax_stage': seq_softmax_stage_})

    #seq_data_ = np.reshape(seq_data_, [-1, steps_of_history, dim])
    #seq_damage_ = np.reshape(seq_damage_, [-1, 1])

    # data statistics
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

    return dd_new #{'seq_data': seq_data_, 'seq_damage': seq_damage_, 'data': data_, 'damage': damage_, 'stage': stage_}



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
    def save(self, sess, name, global_step=None):
        self.network.save(sess, name, global_step=global_step)
    def network_params(self):
        return self.network.network.network_params


class cl_critic(cl):
    def __init__(self, size, config):
        config['cl_structure'] = 'ffcritic:fc_relu_4;fc_relu_3;fc_relu_3'
        super().__init__(size, config)
    def predict(self, sess, s_batch, a_batch):
        return self.network.predict_(sess, s_batch, action=a_batch)
    def train(self, sess, s_batch, y_i, **kwargs):
        self.network.train(sess, s_batch, y_i, **kwargs)


class cl_critic_target(cl):
    def __init__(self, size, config):
        config["cl_target"] = True
        config['cl_structure'] = 'ffcritic:fc_relu_4;fc_relu_3;fc_relu_3'
        super().__init__(size, config)
    def predict(self, sess, s_batch, a_batch):
        return self.network.predict_target_(sess, s_batch, action=a_batch)
    def train(self, sess, s_batch, y_i, **kwargs):
        self.network.train(sess, s_batch, y_i, **kwargs)
        self.network.update_target_network(sess)

class cl_ff_regression(cl):
    def __init__(self, size, config):
        #config['cl_structure'] = 'ffr:fc_relu_;fc_linear_1'
        config['cl_structure'] = 'ffr:fc_linear_1'
        super().__init__(size+1, config) # action included to state
    def predict(self, sess, s_batch, a_batch):
        s_batch_new = np.concatenate((s_batch, a_batch), axis=1)
        return self.network.predict_(sess, s_batch_new)
    def train(self, sess, s_batch, y_i, **kwargs):
        a_batch = kwargs['a_batch']
        s_batch_new = np.concatenate((s_batch, a_batch), axis=1)
        self.network.train(sess, s_batch_new, y_i)

class cl_rnn_classification(cl):
    def __init__(self, size, config):
        #config['cl_structure'] = 'rnnc:rnn_tanh_6_dropout;fc_linear_3'
        config['cl_structure'] = 'rnnc:gru_tanh_6_dropout;fc_linear_3'
        #config['cl_structure'] = 'rnnc:lstm_tanh_6_dropout;fc_linear_3'
        super().__init__(size, config)
    def predict(self, sess, s_batch, a_batch=None):
        return self.network.predict_(sess, s_batch)
    def train(self, sess, s_batch, y_i, **kwargs):
        return self.network.train(sess, s_batch, y_i, **kwargs)


def main():
    params = {}
    params['steps_of_history'] = 2
    params['zero_padding_after'] = 1
    params['damamge_threshold'] = 10000.0
    params['indi_norm'] = True
    params['damage_norm'] = 'to_reward'
    params['stage_norm'] = '' #'cetered'
    params['neg_damage'] = True

    config = parse_args()
    config['default_damage'] = 4035.00
    config["cl_tau"] = 0.001
    config['cl_dropout_keep'] = 0.7
    config["cl_l2_reg"] = 0.001
    config["minibatch_size"] = 128

    random_shuffle = False #True
    test_percentage = 0.3
#    config["cl_lr"] = 0.001
#    training_epochs = 10000
#    export_names = "long_curriculum_network"
    config["cl_lr"] = 0.01
    training_epochs = 1000
    export_names = "short_curriculum_network"
    nn_params = (export_names, "{}_stat.pkl".format(export_names))

    #dd = load_data('leo_supervised_learning_regression2/', params, gens = [2])
    dd = load_data('leo_supervised_learning_regression/', params, gens = range(1,7))

    # Rule-based processing before splitting
    dd = clean_data(dd, params)
    dd = process_data(dd, config)
    dd = select_good(dd, config, 0.1)

    # split into training and testing sets
    idx = np.arange(len(dd))
    if random_shuffle:
        np.random.shuffle(idx)
    test_idx = int(len(dd)*test_percentage)
    dd_train = [d for i,d in enumerate(dd) if i in idx[test_idx:]]
    dd_test = [d for i,d in enumerate(dd) if i in idx[:test_idx]]

    # normalize training dataset and use moments to normalizing test dataset
    dd_train, data_norm, damage_norm = normalize_data(dd_train, config, params)
    dd_test, _, _ = normalize_data(dd_test, config, params, data_norm, damage_norm)

    # save means and std for usage in RL
    config['cl_depth'] = params['steps_of_history']
    config['cl_pt_shape'] = (params['steps_of_history'],dim)
    pt = PerformanceTracker(config,
                            input_norm=data_norm,
                            output_norm=damage_norm)
    pt.save(nn_params[1])

    # get stat
    seq_train = seq_cut(dd_train, params)
    seq_test = seq_cut(dd_test, params)

    # calculate class weights
    #class_weight = calc_class_weight(dd_train, params)

    # fill in replay beuffer
    print(len(seq_train), len(seq_test))
    plot_train, plot_test = [], []
    plt.gca().set_color_cycle(['red', 'green', 'blue'])

    with tf.Graph().as_default() as sl:
        #cl_nn = cl_critic(pt.get_v_size(), config)
        #cl_nn = cl_critic_target(pt.get_v_size(), config)
        cl_nn = cl_rnn_classification(pt.get_v_size(), config)


    display_step = 10
    with tf.Session(graph=sl) as sess:
        # random initialization of variables
        sess.run(tf.global_variables_initializer())

        # Training cycle
        for epoch in range(training_epochs):
            if random_shuffle:
                np.random.shuffle(seq_train)
            avg_loss = 0.
            # Loop over all sequences
            for seq in seq_train:
                # Run optimization op (backprop) and cost op (to get loss value)

                x = np.reshape(seq['seq_data'], [-1, params['steps_of_history'], 3])
                y = np.reshape(seq['seq_softmax_stage'], [-1, 3])

                class_counter = np.sum(y, axis=0)
                class_counter = np.reshape(class_counter, [-1, 1])
                class_weight = (sum(class_counter) - class_counter) / sum(class_counter)
                weights = y.dot(class_weight)
                weights = weights / np.linalg.norm(weights)

                _, loss = cl_nn.train(sess, x, y, class_weight=weights)
                avg_loss += loss

            # Compute average loss
            avg_loss += avg_loss / len(seq_train)

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss))

                if epoch % 100 == 0:
                    plot_train.append(calc_error(sess, cl_nn, seq_train, params))
                    plot_test.append(calc_error(sess, cl_nn, seq_test, params))
                    plot_train_ = np.reshape(plot_train, (-1, 3))
                    plot_test_ = np.reshape(plot_test, (-1, 3))
                    plt.plot(plot_train_, linestyle='-')
                    plt.plot(plot_test_, linestyle=':')

                    plt.pause(0.05)


        print("Optimization Finished!")

        cl_nn.save(sess, nn_params[0])
        plt.show(block=True)


def calc_error(sess, cl_nn, sseq, params):
    plabels = np.zeros(3)
    num_labels = np.zeros(3)
    for seq in sseq:
        xx = np.reshape(seq['seq_data'], [-1, params['steps_of_history'], 3])
        for x, softmaxl in zip(xx,seq['seq_softmax_stage']):
            x = np.reshape(x, [-1, params['steps_of_history'], 3])
            rr = cl_nn.predict(sess, x)
            label = np.argmax(rr)
            softmaxli = np.argmax(softmaxl)
            plabels[softmaxli] += softmaxl[label]
            num_labels[softmaxli] += 1

    accuracy = plabels/num_labels
    for i in range(dim):
        print('accuracy = {} of {}'.format(accuracy[i], num_labels[i]))
    return accuracy


######################################################################################
if __name__ == "__main__":
    main()

