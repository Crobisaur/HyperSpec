from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" A HyperSpectral neural network which displays summaries in TensorBoard.
 Based on an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                                         'for unit testing.')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/home/crob/HyperSpec_logs', 'Summaries directory')

f = h5py.File('/home/crob/HyperSpec/Python/BSQ_test.h5','r')

def convert_labels(labels,n_classes, debug = False):
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



dcb = f['data'][:]
train_dcb = dcb[0:2200000,:]
test_dcb = dcb[3000000::,:]
labels = f['labels'][:]
lambdas = f['bands'][:]
binLabels = f['bin_labels'][:]
train_labels = binLabels[0:2200000,:]
test_labels = binLabels[3000000::,:]
wbc = dcb[labels]
wbc_train = wbc[0:75000,:]
wbc_test= wbc[75000::,:]

#wbc_labels = labels == 1
wbc_labels = np.ones(len(wbc))
wbc_train_labels = wbc_labels[0:75000]
wbc_test_labels = wbc_labels[75000::]

other = dcb[np.invert(labels)]
other_train = other[10::10,:]
other_test = other[7::7,:]
other_labels = np.zeros(len(other))
other_labels_train = other_labels[10::10]
other_labels_test = other_labels[7::7]

combset_train = np.vstack([wbc_train, other_train])
combset_test = np.vstack([wbc_test, other_test])
combset_train_labels = convert_labels(np.hstack([wbc_train_labels, other_labels_train]), 2)
combset_test_labels = convert_labels(np.hstack([wbc_test_labels, other_labels_test]), 2)


#combset = np.vstack([wbc,other[np.random.randint(250000)]])

f.close()

def train():
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True,
                                      #fake_data=FLAGS.fake_data)

    sess = tf.InteractiveSession()

    # Create a multilayer model.
    def conv2d(x, W):

        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    # Input placehoolders
    with tf.name_scope('input'):

        x = tf.placeholder(tf.float32, [None, 25], name='x-input')
        image_shaped_input = tf.reshape(x, [-1, 25])
        tf.image_summary('input', image_shaped_input, 2)
        y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
        keep_prob = tf.placeholder(tf.float32)
        tf.scalar_summary('dropout_keep_probability', keep_prob)

    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            #tf.histogram_summary(name, var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read, and
        adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
           #     tf.histogram_summary(layer_name + '/pre_activations', preactivate)
            activations = act(preactivate, 'activation')
     #       tf.histogram_summary(layer_name + '/activations', activations)
            return activations

    hidden1 = nn_layer(x, 25, 10, 'layer1')
    dropped = tf.nn.dropout(hidden1, keep_prob)
    #hidden2 = nn_layer(dropped, 20, 30,'layer2')
    #dropped2 = tf.nn.dropout(hidden2, keep_prob)
    y = nn_layer(dropped, 10, 2, 'layer2', act=tf.nn.softmax)


    with tf.name_scope('cross_entropy'):
        diff = y_ * tf.log(y)
        with tf.name_scope('total'):
            cross_entropy = -tf.reduce_mean(diff)
        tf.scalar_summary('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(
            FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
    tf.initialize_all_variables().run()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
            xs, ys = train_dcb, train_labels  #.next_batch(100, fake_data=FLAGS.fake_data)  combset_train, combset_train_labels
            k = FLAGS.dropout
        else:
            xs, ys = test_dcb, test_labels #combset_test, combset_test_labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else: # Record train set summarieis, and train
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)

def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    tf.app.run()

