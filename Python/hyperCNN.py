from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import metrics, cross_validation
from PIL import Image
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

#f = h5py.File('/home/crob/HyperSpec/Python/BSQ_test.h5','r')
class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)


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

def convLabels(labelImg, numBands):
    '''
    takes a MxNx3 numpy array and creates binary labels based on predefined classes
    background = 0
    red = 1 WBC
    green = 2 RBC
    pink = 3 nuclear material
    yellow = 4 ignore
    '''


    #b = np.uint8(numBands / 31)
    # print(b / 31)
    tempRed = labelImg[:,:,0] == 255
    tempGreen = labelImg[:,:,1] == 255
    tempBlue = labelImg[:,:,2] == 255
    tempYellow = np.logical_and(tempRed, tempGreen)
    tempPink = np.logical_and(tempRed, tempBlue)
    temp = np.zeros(np.shape(tempRed))
    temp[tempRed] = 1
    temp[tempGreen] = 2
    temp[tempPink] = 3
    temp[tempYellow] = 4
    print(temp)
    print(tempRed, tempGreen, tempBlue, tempYellow, tempPink)
    return temp



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



def getData(filename=None):
    if filename is None: filename = '/home/crob/HyperSpec/Python/BSQ_test.h5'
    f = h5py.File(filename, 'r')
    dcb = f['data'][:]
    labels = f['labels'][:]
    bands = f['bands'][:]
    classLabels = f['classLabels'][:]
    out = {'dcb': dcb, 'labels': labels, 'lambdas': bands, 'classLabels': classLabels}
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

def shapeData(data, labels, numExamples, numBands, altDims = None):
    '''Takes input data matrix and reshapes it into HW,D format
    i.e. endmembers and their appropriate class labels'''
    if altDims is None: altDims = [443, 313, numBands, numExamples]
    temp = np.reshape(data, altDims, 'f')
    dataR = np.reshape(temp,[-1, numBands])
    labelL = np.reshape(labels, [-1,1])
    out = {'data': dataR, 'label': labelL}
    return out

def conv_model(X, y):
    X = tf.expand_dims(X, 3)
    features = tf.reduce_max(tf.contrib.learn.ops.conv2d(X, 12, [3, 3]), [1, 2])
    features = tf.reshape(features, [-1, 12])
    return tf.contrib.learn.models.logistic_regression(features, y)

if __name__ == '__main__':
    trainData = getData(filename='/home/crob/HYPER_SPEC_TRAIN.h5')
    testData = getData(filename='/home/crob/HYPER_SPEC_TEST.h5')
    a = np.shape(trainData['dcb'])
    b = np.uint8(a[2]/31)
    print(b / 31)
    #lab = np.reshape(testData['labels'], [443,313,3,b],'f')
    #numExamples = np.shape(lab)
    #for j in range(np.uint8(numExamples[3])):
    #   a = convLabels(lab[:,:,:,j], None)



    # working on reshaping images into w*h,d format
    nn = np.reshape(trainData['dcb'],[443,313,31,138],'f')
    #no need for fortran encoding this time
    c = np.reshape(nn[:,:,:,1],[443*313,31])
    print(np.shape(c))
    d = np.reshape(trainData['classLabels'],[443,313,138],'f')
    dd = np.reshape(d[:,:,1],[443*313])

    train = shapeData(trainData['dcb'], trainData['classLabels'], 138, 31)
    test = shapeData(testData['dcb'], testData['classLabels'], 12, 31)
    print(train['data'])
    print(train['label'])
    print(test['data'])
    print(test['label'])

#    c = lab[:,:,:,1]d
    #print(np.shape(c))
    ##plt.figure(1)
    fig, ax = plt.subplots()
    ##plt.subplot(311)
    img = ax.imshow(c[0:313*2,:])#c[:,:,0])
    ##fig.add_subplot(312)
    ##plt.imshow(c[:,:,1])
    ##fig.add_subplot(313)
    ##plt.imshow(c[:,:,2])
    ax.format_coord = Formatter(img)
    #print(dd[0:313 * 2])
    #plt.show()


    #print data structure for reference
    print(a)

    print(np.reshape(trainData['dcb'], [443,313,31,138],'f'))
    print(trainData['dcb'])
    print(trainData['classLabels'])
    # reshape into vectors for easy data loading
    # make sure labels are in same order

    val_monitor = tf.contrib.learn.monitors.ValidationMonitor(test['data'], test['label'], every_n_steps=50)


    classifier = tf.contrib.learn.TensorFlowDNNClassifier(hidden_units=[50,100,80,50],
                                               n_classes=5, steps=20000, learning_rate=0.05)

    #iris = tf.contrib.learn.datasets.load_dataset('iris')
    #print(iris.data)
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(test['data'], np.uint8(test['label']),
    #                                                                     test_size=0.3, random_state=42)
    print(np.shape(train['data']))
    print(np.shape(train['label']))
    classifier.fit(train['data'], train['label'], val_monitor)
    classifier.save('testModel1/')
    score = metrics.accuracy_score(test['label'], classifier.predict(test['data']))


    print('Test Accuracy: {0:f}'.format(score))
