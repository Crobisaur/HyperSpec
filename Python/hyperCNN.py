from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""A HS Convolutional Neural Network that detects if a leukocyte is present in the image.
   Let's see how this pans out...
"""
#set dims of input image

dcb_size = {'height': 443, 'width': 313,
            'channel_25': 25, 'channel_31': 31}

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit test')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run training')
flags.DEFINE_float('learning_rate', 0.001, 'Initial Learning Rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/home/crob/HyperSpec_logs',
                    'Directory for storing tensorboard summaries.')

f = h5py.File('/home/crob/HyperSpec/Python/BSQ_test.h5','r')

def convert_labels(labels, n_classes, debug = False):
    for j in range(n_classes):
        temp = labels == j
        temp = temp.astype(int)
        if j > 0:
            conv_labels = np.append(conv_labels, temp)
            print(temp[:])
        else:
            conv_labels = temp
    print(np.shape(conv_labels))
    conv_labels = np.reshape(conv_labels, [len(labels), n_classes], order='F')
    if debug: print(np.shape(conv_labels))
    return conv_labels

def getData(filename=None):
    if filename is None: filename = '/home/crob/HyperSpec/Python/BSQ_test.h5'
    f = h5py.File(filename, 'r')
    dcb = f['data'][:]
    labels = f['labels'][:]
    bands = f['bands'][:]
    out = {'dcb': dcb, 'labels': labels, 'lambdas': bands}
    return out


def read_data_as_vect(filepath, debug = False):
    '''reads in data from hdf5 file as a list of vectors'''
    f = h5py.File('/home/crob/HyperSpec/Python/BSQ_test.h5', 'r')
    dcb = f['data'][:]
    dcb = f['data'][:]
    train_dcb = dcb[0:2200000, :]
    test_dcb = dcb[3000000::, :]
    labels = f['labels'][:]
    lambdas = f['bands'][:]
    binLabels = f['bin_labels'][:]
    train_labels = binLabels[0:2200000, :]
    test_labels = binLabels[3000000::, :]
    wbc = dcb[labels]
    wbc_train = wbc[0:75000, :]
    wbc_test = wbc[75000::, :]

    # wbc_labels = labels == 1
    wbc_labels = np.ones(len(wbc))
    wbc_train_labels = wbc_labels[0:75000]
    wbc_test_labels = wbc_labels[75000::]

    other = dcb[np.invert(labels)]
    other_train = other[10::10, :]
    other_test = other[7::7, :]
    other_labels = np.zeros(len(other))
    other_labels_train = other_labels[10::10]
    other_labels_test = other_labels[7::7]

    combset_train = np.vstack([wbc_train, other_train])
    combset_test = np.vstack([wbc_test, other_test])
    combset_train_labels = convert_labels(np.hstack(
        [wbc_train_labels, other_labels_train]), 2)
    combset_test_labels = convert_labels(np.hstack(
        [wbc_test_labels, other_labels_test]), 2)
    f.close()

def cnnTrain():
    #Import data & create placeholders
    def conv2d(img, w, b):
        '''A convolutional Layer which adds in biases'''
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
            img, w, strides=[1,1,1,1], padding='SAME'), b))

    def max_pool(img, k):
        '''A max pooling Layer for k size pooling'''
        return tf.nn.max_pool(
            img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv_net(_X, _weights, _biases, _dropout):
        # Reshape Input
        # This is our datacube input
        _X = tf.reshape(_X, shape=[-1, dcb_size('height'), dcb_size('width'), dcb_size('channel_25')])