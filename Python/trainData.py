__author__ = "Christo Robison"
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
import time

class trainData:
    """Object which stores training and test datasets"""
    def __init__(self, traindata=None, trainlabels=None, testdata=None, testlabels=None):
        self.trainData = traindata
        self.testData = testdata
        self.trainLabels = trainlabels
        self.testLabels = testlabels
        self.cacheSize = 2048  # size of cache for SVM in MB
        if self.trainData is not None:
            self.muTrain = np.mean(self.trainData)
            self.sigTrain = np.std(self.trainData)
        else:
            self.muTrain, self.sigTest = None
        if self.testData is not None:
            self.muTest = np.mean(self.testData)
            self.sigTest = np.std(self.testData)
        else:
            self.muTest, self.sigTest = None

    def kernelSAM(self, T, R):
        """
        Spectral Angle Mapper eqn between test and reference vectors T & R respectively
        """
        return np.arccos(np.dot(T,R)/(np.linalg.norm(T)*np.linalg.norm(R)))

    def trainSVM(self, kernel=None):
        if kernel is None:
            kernel = 'linear'
        elif kernel.lower() == 'sam':
            kernel = self.kernelSAM
        clf = svm.SVC(kernel=kernel(self.trainData, self.))
        clf.fit()





