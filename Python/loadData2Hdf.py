__author__ = "Christo Robison"

import numpy as np
import h5py
from PIL import Image
import os
import collections
import convertBsqMulti as bsq

'''This program reads in BSQ datacubes into an HDF file'''

def loadBSQ(path = '/home/crob/HyperSpec_Data/WBC v ALL/WBC25', debug=False):
    d31 = []
    d25 = []
    l25 = []
    l = []
    lam = []
    for root, dirs, files in os.walk(path):
        for name in sorted(files): #os walk iterates arbitrarily, sort fixes it
            print(name)
            if name.endswith(".png"):
                # Import label image
                im = np.array(Image.open(os.path.join(root,name)),'f')
                print np.shape(im)
                bw = im[:,:,0] > 250
                print(np.shape(bw))
                l.append(bw)
                #l.append(np.reshape(bw, 138659))

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
                    d31.append(np.reshape(np.transpose(bs[0], (1, 2, 0)), 4298429))

                if len(bs[1]) == 25:
                    print('BSQ is size 25')
                    print(len(bs[1]))
                    lam = bs[1]
                    d25.append(np.reshape(np.transpose(bs[0], (1, 2, 0)), 3466475))

    out = collections.namedtuple('examples',['data31','data25', 'labels', 'lambdas'])
    o = out(data31=d31, data25=d25, labels=l, lambdas=lam)
    return o


if __name__ == '__main__':
    A = loadBSQ()
    path = '/home/crob/HyperSpec_Data/WBC v ALL/WBC25'
    s = loadBSQ(path)
    print(np.shape(s.data25))
    f = h5py.File("BSQ_test.h5", "w")
    f.create_dataset('data', data=s.data25)
    f.create_dataset('labels', data=s.labels)
    f.create_dataset('bands', data=s.lambdas)
    f.close()