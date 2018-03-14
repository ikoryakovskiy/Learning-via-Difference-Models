from __future__ import division
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

from ptracker import PerformanceTracker
from cl_network import CurriculumNetwork
from ddpg import parse_args


def main():
    config = parse_args()

    with open("leo_supervised_learning/positive_keys.txt", 'r') as f:
        fpositives = f.readlines()
    fpositives = [x.strip() for x in fpositives]

    with open("leo_supervised_learning/negative_keys.txt", 'r') as f:
        fnegatives = f.readlines()
    fnegatives = [x.strip() for x in fnegatives]

    # now create two lists of positive and negative training samples
    config['reach_return'] = 1422.66
    config['cl_structure'] = 'softmax_2' # softmax classifier for a number of classes
    config['cl_depth'] = 2
    config["cl_l2_reg"] = 0.01

    pt = PerformanceTracker(depth=config['cl_depth'], dim=4, input_norm=config["cl_input_norm"])
    with tf.Graph().as_default() as sa:
        cl_nn = CurriculumNetwork(pt.get_v_size(), config)

    N = None
    P = None
    for fps in fpositives:
        fbalancing, fwalking = fps.split()

        data = np.loadtxt(fbalancing, skiprows=2, usecols=(1, 3, 4))
        rw = data[:, 0, np.newaxis]/config['reach_return']  # return
        tt = data[:, 1, np.newaxis]/config['env_timeout']   # duration
        fl = data[:, 2, np.newaxis]                         # falls
        fl_rate = np.diff(np.vstack(([0], fl)), axis=0) / config["test_interval"]
        labels  = np.hstack((np.zeros(rw.shape), np.ones(rw.shape)))
        labels[-1, :] = np.array([1, 0]) # positive label in the end of the sequence
        normalized_data = np.hstack((rw,tt,fl_rate))

        # add two rows of zeros which correspond to situations when PerformanceTracker is empty
        normalized_data = np.vstack((np.zeros(shape=(2,3)),normalized_data))
        labels = np.vstack((np.zeros(shape=(2,2)),labels))
        reshaped_data = np.hstack((normalized_data[:-1, :], normalized_data[1:, :], labels[1:,:]))

        neg_data = reshaped_data[:-1, :]
        pos_data = reshaped_data[ -1, :]
        N = np.vstack((N, neg_data)) if N is not None else neg_data
        P = np.vstack((P, pos_data)) if P is not None else pos_data

    for fps in fnegatives:
        fbalancing, fwalking = fps.split()

        data = np.loadtxt(fbalancing, skiprows=2, usecols=(1, 3, 4))
        rw = data[:, 0, np.newaxis]/config['reach_return']  # reward
        tt = data[:, 1, np.newaxis]/config['env_timeout']   # duration
        fl = data[:, 2, np.newaxis]                         # falls
        fl_rate = np.diff(np.vstack(([0], fl)), axis=0) / config["test_interval"]
        labels  = np.hstack((np.zeros(rw.shape), np.ones(rw.shape)))
        normalized_data = np.hstack((rw,tt,fl))

        # add two rows of zeros which correspond to situations when PerformanceTracker is empty
        normalized_data = np.vstack((np.zeros(shape=(2,3)),normalized_data))
        labels = np.vstack((np.zeros(shape=(2,2)),labels))
        reshaped_data = np.hstack((normalized_data[:-1, :], normalized_data[1:, :], labels[1:,:]))

        N = np.vstack((N, neg_data))

    # divide into training and testing sets
    testing_percentage = 0.7
    np.random.shuffle(N)
    np.random.shuffle(P)
    trN = N[0:int(N.shape[0]*testing_percentage), :]
    tsN = N[int(N.shape[0]*testing_percentage):, :]
    trP = P[0:int(P.shape[0]*testing_percentage), :]
    tsP = P[int(P.shape[0]*testing_percentage):, :]

    training_epochs = 10000
    batch_size = 10
    display_step = 10
    with tf.Session(graph=sa) as sess:
        # random initialization of variables
        sess.run(tf.global_variables_initializer())

        # Training cycle
        for epoch in range(training_epochs):
            np.random.shuffle(trN)
            trN_sub = trN[0:trP.shape[0]]
            tr = np.vstack((trN_sub, trP))
            np.random.shuffle(tr)
            batches_num = tr.shape[0] / batch_size
            avg_cost = 0.
            # Loop over all batches
            for batch in np.array_split(tr, batches_num):
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = cl_nn.train(sess, batch[:, 0:6], batch[:, 6:8])

                # Compute average loss
                avg_cost += c / batch.shape[0]
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Sensitivity / Specificity check
        nlabels, plabels = [], []

        for v in tsP:
            plabels.append(cl_nn.predict(sess, v[np.newaxis, 0:6])) # output should be 1 for positives
        for v in tsN:
            nlabels.append(cl_nn.predict(sess, v[np.newaxis, 0:6])) # output should be 0 for negatives

        #fp = [tsN[i, :] for i, l in enumerate(nlabels) if l == 1]

        TP = sum(plabels) / len(plabels)
        FN = 1 - TP
        FP = sum(nlabels) / len(nlabels)
        TN = 1 - FP
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        print('sensitivity = {}, specificity = {}'.format(sensitivity, specificity))

        params = cl_nn.get_params(sess)
        print('params = {}'.format(params))

        cmaes_params = params[::2] - params[1::2]
        print('cmaes_param = {}'.format(cmaes_params))

        pass




######################################################################################
if __name__ == "__main__":
    main()

