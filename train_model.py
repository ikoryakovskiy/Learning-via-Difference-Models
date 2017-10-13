#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import pickle
import numpy as np
import os
import yaml

def get_training_data(file_perturbed, file_original):

    points = 5000
    files_num = len(file_perturbed)
    data_points = points*files_num
    training_data_points = int(0.75*data_points)
    set_train = int(training_data_points/files_num)
    set_test = points - set_train
    test_data_points = data_points - training_data_points
    b1_train = np.zeros((training_data_points, 42))
    b1_test = np.zeros((test_data_points, 42))
    b2_train = np.zeros((training_data_points, 42))
    b2_test = np.zeros((test_data_points, 42))
    counter = 0

    for ii in file_perturbed:

        with open(ii) as f:
            transitions = pickle.load(f)
            f.close()
            s = np.array([_[0] for _ in transitions])
            a = np.array([_[1] for _ in transitions])
            s2 = np.array([_[4] for _ in transitions])
            b1_train[set_train*counter:set_train*(counter+1),:] = np.concatenate((s[0:set_train,:], a[0:set_train,:], s2[0:set_train,:]), axis=1)
            b1_test[set_test*counter:set_test*(counter+1),:] = np.concatenate((s[set_train:points,:], a[set_train:points,:], s2[set_train:points,:]), axis=1)
            counter += 1
            # b1 = np.vstack((b1, tmp))

    counter = 0
    for ii in file_original:
        with open(ii) as f:
            transitions = pickle.load(f)
            f.close()

            s = np.array([_[0] for _ in transitions])
            a = np.array([_[1] for _ in transitions])
            s2 = np.array([_[4] for _ in transitions])
            b2_train[set_train*counter:set_train*(counter+1),:] = np.concatenate((s[0:set_train,:], a[0:set_train,:], s2[0:set_train,:]), axis=1)
            b2_test[set_test*counter:set_test*(counter+1),:] = np.concatenate((s[set_train:points,:], a[set_train:points,:], s2[set_train:points,:]), axis=1)
            counter += 1

    return b1_train, b1_test, b2_train, b2_test


def training(sess, model, file_perturbed, file_original, input_dim, output_dim, epoch = 300):
    check_for_diff_model = False
    path = os.path.dirname(os.path.abspath(__file__))
    yfile = '{}/config.yaml'.format(path)
    if not os.path.isfile(yfile):
        print ('File %s not found' % yfile)
    else:
        stream = file(yfile, 'r')
        conf = yaml.load(stream)
        stream.close()
        diff_model = int(conf['difference_model'])
        if diff_model:
            print ("It goes in!")
            check_for_diff_model = True

    # Initialize neural net
    # model = DifferenceModel(input_dim, output_dim)

    # Load data
    data_perturbed_train, data_perturbed_test, data_original_train, data_original_test = get_training_data(file_perturbed, file_original)
    training_size_data_set = data_perturbed_train.shape[0]
    batch_size = 32
    batches = int(training_size_data_set / batch_size)
    print(training_size_data_set, batch_size, batches)

    # Compose training data
    state = data_perturbed_train[:, 0:output_dim]
    state[:, 0] = 0
    action = np.reshape(data_perturbed_train[:, output_dim:input_dim],
                        (data_perturbed_train.shape[0], input_dim - output_dim))
    next_state_perturbed_train = data_perturbed_train[:, input_dim:input_dim + output_dim]
    next_state_original_train = data_original_train[:, input_dim:input_dim + output_dim]
    diff_state_train = next_state_perturbed_train - next_state_original_train
    training_data = np.concatenate((state, action, diff_state_train), axis=1)

    # Compose test data
    state = data_perturbed_test[:, 0:output_dim]
    state[:, 0] = 0
    action = np.reshape(data_perturbed_test[:, output_dim:input_dim],
                        (data_perturbed_test.shape[0], input_dim - output_dim))
    next_state_perturbed_test = data_perturbed_test[:, input_dim:input_dim + output_dim]
    next_state_original_test = data_original_test[:, input_dim:input_dim + output_dim]
    diff_state_test = next_state_perturbed_test - next_state_original_test
    test_data = np.concatenate((state, action, diff_state_test), axis=1)

    # Check for difference model
    if check_for_diff_model:
        saver = tf.train.Saver()
        saver.restore(sess, "./difference-model")
        print ('Difference model restored for training')
    else:
        sess.run(tf.global_variables_initializer())

    print(training_data.shape)

    tmp = training_data.copy()
    # Train the model
    for _ in range(epoch):
        np.random.shuffle(tmp)
        for i in range(batches):
            model.train(sess, tmp[batch_size*i:batch_size*(i+1),0:input_dim], tmp[batch_size*i:batch_size*(i+1),input_dim:input_dim+output_dim])

    # Save the model
    saver = tf.train.Saver()
    saver.save(sess, "difference-model")

    # Printing error results
    next_state_predicted_output_test = model.predict(sess, test_data[:, 0:input_dim], 1) + next_state_original_test
    print ("Original error on validation set", np.mean(((next_state_original_test - next_state_perturbed_test) ** 2)))
    print ("Error on validation Set", np.mean(((next_state_predicted_output_test - next_state_perturbed_test) ** 2)))

    next_state_predicted_output_train = model.predict(sess, training_data[:, 0:input_dim], 1) + next_state_original_train
    print ("Original error on validation set", np.mean(((next_state_original_train - next_state_perturbed_train) ** 2)))
    print ("Error on training Set", np.mean(((next_state_predicted_output_train - next_state_perturbed_train) ** 2)))

    return model


def prediction(sess, model, inputs, input_dim, output_dim):

    # Predict using the model
    state = inputs[:, 0:output_dim]
    state[:,0] = 0
    action = np.reshape(inputs[:, output_dim:input_dim], (inputs.shape[0], input_dim-output_dim))
    next_state = inputs[:, input_dim:input_dim + output_dim]
    state_output = model.predict(sess, np.concatenate((state, action), axis=1), 1) + next_state

    return state_output


