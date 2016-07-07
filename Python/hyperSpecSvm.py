__author__='Christo Robison'

import numpy as np
from sklearn import svm
import h5py
import time


runSAM = False

def kernelSAM(T, R):
    """
    Spectral Angle Mapper eqn between test and reference vectors T & R respectively
    """
    return np.arccos(np.dot(T, R) / (np.linalg.norm(T,axis=1) * np.linalg.norm(R)))

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

def getClassExamples(data, classNum):
    kee = np.equal(data['label'],classNum)
    out = data['data']*kee
    return out

def getClassMean(data, classNum):
    kee = np.equal(data['label'],classNum)
    out = np.mean(data['data']*kee,axis=0)
    return out

def getClassExamples(data, classNum):
    kee = np.equal(data['label'],classNum)


    out = (data['data']*kee)
    out=out[~(out==0).all(1)]
    ran = np.random.permutation(len(out))
    ran = ran[:2500]
    out = out[ran]
    return out

def getAverages(data, numClasses):
    out = []
    for i in range(numClasses):
        a = getClassMean(data, i)
        out.append(a)
    return out

def sam_Classes(data, avg):
    t = []
    for i in range(len(avg)):
        t.append(kernelSAM(data,avg[i]))
    return t

if __name__ == '__main__':
    trainData = getData(filename='/home/crob/HYPER_SPEC_TRAIN.h5')
    testData = getData(filename='/home/crob/HYPER_SPEC_TEST.h5')
    a = np.shape(trainData['dcb'])
    b = np.uint8(a[2] / 25)
    print(b / 25)  # This needs fixed for when cubes are 25 or 31 bands
    # lab = np.reshape(testData['labels'], [443,313,3,b],'f')
    # numExamples = np.shape(lab)
    # for j in range(np.uint8(numExamples[3])):
    #   a = convLabels(lab[:,:,:,j], None)



    # working on reshaping images into w*h,d format
    nn = np.reshape(trainData['dcb'], [443, 313, 25, 380], 'f')
    # no need for fortran encoding this time
    c = np.reshape(nn[:, :, :, 1], [443 * 313, 25])
    print(np.shape(c))
    d = np.reshape(trainData['classLabels'], [443, 313, 380], 'f')
    dd = np.reshape(d[:, :, 1], [443 * 313])

    train = shapeData(trainData['dcb'], trainData['classLabels'], 380, 25)  # 138 for old data set
    test = shapeData(testData['dcb'], testData['classLabels'], 30, 25)  # 12 for old test set
    print(train['data'])
    print(train['label'])
    print(test['data'])
    print(test['label'])

    batch = {'data': train['data'][50000], 'label': train['label'][50000]}

    if runSAM is True:
        p = getAverages(train, 5)
        sam_results = []
   # for i in range(len(train['data'])):
        sam_results = sam_Classes(train['data'],p)
        sam_results = np.rot90(np.reshape(np.transpose(sam_results),[443,313,5,-1]),2)
        print(p)
        print(np.shape(sam_results))
        f = h5py.File('/home/crob/HYPER_SPEC_TRAIN.h5','r+')
        f.create_dataset('class_avg',data=p)
        f.create_dataset('sam',data=sam_results)
        f.close()

    TrainSamplesClass0 = getClassExamples(train, 0)
    TrainLabelsClass0 = np.zeros(2500)

    TrainSamplesClass1 = getClassExamples(train, 1)
    TrainLabelsClass1 = np.zeros(2500)
    TrainLabelsClass1.fill(1)

    TrainSamplesClass2 = getClassExamples(train, 2)
    TrainLabelsClass2 = np.zeros(2500)
    TrainLabelsClass2.fill(2)

    TrainSamplesClass3 = getClassExamples(train, 3)
    TrainLabelsClass3 = np.zeros(2500)
    TrainLabelsClass3.fill(3)

    TrainSamplesClass4 = getClassExamples(train, 4)
    TrainLabelsClass4 = np.zeros(2500)
    TrainLabelsClass4.fill(4)

    trainD = np.concatenate([TrainSamplesClass0, TrainSamplesClass1, TrainSamplesClass2, TrainSamplesClass3, TrainSamplesClass4], axis=0)
    trainL = np.concatenate([TrainLabelsClass0, TrainLabelsClass1, TrainLabelsClass2, TrainLabelsClass3, TrainLabelsClass4], axis=0)
    clf = svm.SVC(cache_size=7000)
    start = time.clock()
    clf.fit(trainD, trainL)
    calc_time = time.clock() - start
    wall_time = time.time() - start
    print([calc_time, wall_time])
    #dec_Fx = clf.decision_function(train['data'])
    pred_results = clf.predict(test['data'][:350000,:])
    print(pred_results)