__author__ = 'Christo Robison'

from spectral import *
import numpy as np
#import tensorflow as tf
import h5py
import pylab
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import fftpack
import time
from skimage import io, exposure, img_as_uint, img_as_float
import png
io.use_plugin('freeimage')



output = r'H:\Results'

def getData(filename=None):
    if filename is None: filename = 'D:\-_Hyper_Spec_-\HYPER_SPEC_TEST.h5'
    f = h5py.File(filename, 'r')
    dcb = f['data'][:] #Extract normalized data for svm b/c intensity sensitive
    labels = f['labels'][:]
    bands = f['bands'][:]
    classLabels = f['classLabels'][:]
    out = {'dcb': dcb, 'labels': labels, 'lambdas': bands, 'classLabels': classLabels}
    f.close()
    return out

def shapeData(data, labels, numExamples, numBands, altDims = None):
    '''Takes input data matrix and reshapes it into HW,D format
    i.e. endmembers and their appropriate class labels'''
    if altDims is None: altDims = [443, 313, numBands, numExamples]
    temp = np.reshape(data, altDims, 'f')
    dataR = np.reshape(temp,[-1, numBands])
    labelL = np.reshape(labels, [-1,1])
    out = {'data': dataR, 'label': labelL}
    return out

if __name__ == '__main__':
    trainData = getData(filename='D:\-_Hyper_Spec_-\HYPER_SPEC_TEST.h5')
    testData = getData(filename='D:\-_Hyper_Spec_-\HYPER_SPEC_TEST_RED.h5')
    print(np.shape(trainData['dcb']))
    for i in range(np.shape(trainData['dcb'])[2]):
        im = exposure.rescale_intensity(trainData['dcb'][:,:,i], out_range='float')
        im = img_as_uint(im)
        io.imsave((r'HYPER_SPEC_TEST\band_image_' + str(i) + '.png'), im)

        #pf = open(('band_image_' + str(i) + '.png'), 'wb')
        #w = png.Writer(width=313, height=443, bitdepth=16, greyscale=True)
       # w.write(pf, np.reshape(testData['dcb'], (-1, 443 * 372)))
        #pf.close()

    ### Unsupervised Classification
    #img = trainData['dcb'][:,:,1625:1651]
    #(m, c) = kmeans(img, 6, 300)
    img = trainData['dcb'][:,:,341:372]
    (m, c) = kmeans(img, 6, 300)
    fig1 = plt.figure(1)
    fig1.hold(True)
    ax1 = fig1.add_subplot(111)
    for i in range(c.shape[0]):
        ax1.plot(c[i])
    #plt.ion()
    #pylab.show()
    fig1.savefig('kmeans')
    fig1.hold(False)
    ####Supervised Classification
    gt = trainData['classLabels'][:,:,11]
    bkgnd = gt == 0
    gt[bkgnd] = 6
    #pylab.figure()
    fig2 = plt.figure(2)
    fig2.hold(True)
    ax2 = fig2.add_subplot(111)

    v = imshow(classes=gt, fignum=None)
    plt.savefig('ground_truth')
    #plt.show()
    #pylab.hold(1)

    classes = create_training_classes(img, gt)

    ###Gaussian Maximum Likelihood Classifier
    gmlc = GaussianClassifier(classes, min_samples=200)
    clmap = gmlc.classify_image(img)
    #pylab.figure()

    v = imshow(classes=clmap)
    plt.savefig('c_map')
    #pylab.hold(1)
    gtresults = clmap * (gt !=0)
    #pylab.figure()

    v = imshow(classes=gtresults)
    plt.savefig('gtresults')
    #pylab.hold(1)
    #pylab.figure()

    gterrors = gtresults * (gtresults != gt)
    v = imshow(classes=gterrors)
    plt.savefig('gterrors')
    #pylab.hold(1)
    #pylab.figure()

    #F1 = fftpack.fft2(img)
    #F2 = fftpack.fftshift(F1)
    #psd2D = np.abs(F2)**2
    F1 = np.fft.rfft2(img)

    v = imshow(F1)
    plt.savefig('fft2')

    pc = principal_components(img)
    v = imshow(pc.cov)
    plt.savefig('covariance_matrix')
    pc_0999 = pc.reduce(fraction=0.999)

    len(pc_0999.eigenvalues)

    img_pc = pc_0999.transform(img)

    v = imshow(img_pc[:,:,:3], stretch_all=True)
    plt.savefig('top3components')
    classes = create_training_classes(img_pc, gt)
    gmlc = GaussianClassifier(classes)
    clmap = gmlc.classify_image(img_pc)
    clmap_training = clmap * (gt !=0)

    v = imshow(classes=clmap_training)
    plt.savefig('trainnigDataC_map')
    training_errors = clmap_training * (clmap_training != gt)
    v = imshow(classes= training_errors)
    plt.savefig('trainingDataErrors')


    #pylab.show()
    time.sleep(1.5)




    #k = input('press to close')