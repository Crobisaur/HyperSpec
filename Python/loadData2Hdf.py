__author__ = "Christo Robison"

import numpy as np
from scipy import signal
from scipy import misc
import h5py
from PIL import Image
import os
import collections
import matplotlib.pyplot as plt
import convertBsqMulti as bsq
import png


'''This program reads in BSQ datacubes into an HDF file'''

def loadBSQ(path = '/home/crob/HyperSpec_Data/WBC v ALL/WBC25', debug=False):
    d31 = []
    d25 = []
    l25 = []
    l = []
    l3 = []
    lam = []
    for root, dirs, files in os.walk(path):
        print(dirs)
        for name in sorted(files): #os walk iterates arbitrarily, sort fixes it
            print(name)
            if name.endswith(".png"):
                # Import label image
                im = np.array(Image.open(os.path.join(root,name)),'f')
                print np.shape(im)
                im = im[:,:,0:3] # > 250
                # generate a mask for 3x3 conv layer (probably not needed)
                #conv3bw = signal.convolve2d(bw, np.ones([22,22],dtype=np.int), mode='valid') >= 464
                print(np.shape(im))
                #p = open(name+'_22sqMask.png','wb')
                #w = png.Writer(255)
                #bw = np.flipud(bw)
                im = np.flipud(im)
                #l3.append(np.reshape(conv3bw, ))
                #l.append(np.reshape(bw, 138659))

                l.append(im)

                print(np.shape(im))
                print("Name = " + name)
            if name.endswith(".bsq"):
                bs = bsq.readbsq(os.path.join(root,name))
                print(np.shape(bs[0]))
                print(len(bs[1]))
                #separate bsq files by prism
                if len(bs[1]) == 31:
                    print('BSQ is size 31')
                    print(len(bs[1]))
                    lam = bs[1]
                    #d31.append(np.reshape(np.transpose(bs[0], (1, 2, 0)), 4298429))
                    d31.append(bs[0].astype(np.float32))


                if len(bs[1]) == 25:
                    print('BSQ is size 25')
                    print(len(bs[1]))
                    lam = bs[1]

                    d25.append(bs[0].astype(np.float32))
                    #d25.append(np.reshape(bs[0],[138659,25]).astype(np.float32))
                    # old don't use #d25.append(np.reshape(np.transpose(bs[0], (1, 2, 0)), 3466475))

    out = collections.namedtuple('examples',['data31','data25', 'labels', 'lambdas'])
    o = out(data31=np.dstack(d31), data25=d25, labels=np.dstack(l), lambdas=lam)  #np.vstack(d25), labels=np.hstack(l)
    return o


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
    if debug:
        f = h5py.File("/home/crob/HyperSpec/Python/BSQ_whole.h5", "w")
        f.create_dataset('bin_labels', data=conv_labels)
        f.close()
    return conv_labels



if __name__ == '__main__':
    #A = loadBSQ()
    path = '/home/crob/-_PreSortedData_Test_-' #oldpath=/HyperSpec_Data/WBC v ALL/WBC25
    s = loadBSQ(path)
    print(np.shape(s.data25))
    f = h5py.File("HYPER_SPEC_TEST.h5", "w")
    f.create_dataset('data', data=s.data31, chunks=(443, 313, 1))
    f.create_dataset('labels', data=s.labels)
    f.create_dataset('bands', data=s.lambdas)
    g = np.shape(s.data31)
    b = np.uint8(g[2] / 31)
    lab = np.reshape(s.labels, [443, 313, 3, b], 'f')
    numExamples = np.shape(lab)
    a = []
    for j in range(np.uint8(numExamples[3])):
        a.append(convLabels(lab[:, :, :, j], None))
    f.create_dataset('classLabels', data=np.dstack(a))
    #p = convert_labels(s.labels,2)
    #f.create_dataset('bin_labels', data=p)
    f.close()