__author__ = "Christo R"

import numpy as np
import h5py


def createTable(filename):
    f = h5py.File(filename, 'w')
    return f

def openTable(filename):
    '''Opens file for read/write and creates if not exist'''
    f = h5py.File(filename, 'a')
    return f

def build25bandGroup(hFile,string=None):
    group = hFile.create_group(string)
    return group

def buildGroup(hFile, string=None):
    group = hFile.create_group(string)
    return group

def addDataset(group, name, data, debug=False):
    """Takes a numpy array and adds it to a dataset in group"""
    d = data
    if debug: print(np.shape(d))
    if debug: print(name)

    dSet = group.create_dataset(name, data=d)
    return dSet

def saveClose(hFile):
    hFile.flush()
    hFile.close()
    return True

if __name__=="__main__":
    data = np.array([[[1,2,4], [233,12,1]],[[1,2,4],[2334,12,1]]])
    print("Testing File Creation")
    HDF=createTable("MYTESTFILE.h5")
    group25=build25bandGroup(HDF,'Prism25_Data_Cube')
    t = group25.create_dataset("THIS",None,None,data)
    saveClose(HDF)