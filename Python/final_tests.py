__author__ = 'Christo Robison'

from spectral import *
import numpy as np
#import tensorflow as tf
import h5py
import pylab
from scipy import fftpack
import time



def getData(filename=None):
    if filename is None: filename = '/home/crob/HyperSpec/Python/BSQ_test.h5'
    f = h5py.File(filename, 'r')
    dcb = f['norm_data'][:] #Extract normalized data for svm b/c intensity sensitive
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
    trainData = getData(filename='/home/crob/HYPER_SPEC_TRAIN.h5')
    testData = getData(filename='/home/crob/HYPER_SPEC_TEST.h5')
    print(np.shape(trainData['dcb']))

    ### Unsupervised Classification
    img = trainData['dcb'][:,:,1625:1651]
    (m, c) = kmeans(img, 6, 300)
    pylab.figure()
    pylab.hold(1)
    for i in range(c.shape[0]):
        pylab.plot(c[i])
    pylab.ion()
    #pylab.show()

    ####Supervised Classification
    gt = trainData['classLabels'][:,:,65]
    #pylab.figure()
    v = imshow(classes=gt)
    pylab.hold(1)

    classes = create_training_classes(img, gt)

    ###Gaussian Maximum Likelihood Classifier
    gmlc = GaussianClassifier(classes, min_samples=200)
    clmap = gmlc.classify_image(img)
    #pylab.figure()

    v = imshow(classes=clmap)
    pylab.hold(1)
    gtresults = clmap * (gt !=0)
    #pylab.figure()

    v = imshow(classes=gtresults)
    pylab.hold(1)
    #pylab.figure()

    gterrors = gtresults * (gtresults != gt)
    v = imshow(classes=gterrors)
    pylab.hold(1)
    #pylab.figure()

    #F1 = fftpack.fft2(img)
    #F2 = fftpack.fftshift(F1)
    #psd2D = np.abs(F2)**2
    F1 = np.fft.rfft2(img)

    v = imshow(F1)


    pc = principal_components(img)
    v = imshow(pc.cov)
    pc_0999 = pc.reduce(fraction=0.999)

    len(pc_0999.eigenvalues)

    img_pc = pc_0999.transform(img)

    v = imshow(img_pc[:,:,:3], stretch_all=True)

    classes = create_training_classes(img_pc, gt)
    gmlc = GaussianClassifier(classes)
    clmap = gmlc.classify_image(img_pc)
    clmap_training = clmap * (gt !=0)

    v = imshow(classes=clmap_training)

    training_errors = clmap_training * (clmap_training != gt)
    v = imshow(classes= training_errors)



    pylab.show()
    time.sleep(1.5)




    #k = input('press to close')