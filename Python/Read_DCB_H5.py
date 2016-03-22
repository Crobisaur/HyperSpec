__author__ = 'Christo Robison'

import dcb_HDF5
import glob, os, sys
import getopt
import numpy as np
import h5py as HDF
from sklearn import cross_validation
from sklearn import svm
import time
import matplotlib

#matplotlib.use('Qt4Agg')

# apparently you can only pass helper functions into f.visit(*helper*)
def printName(name):
    print(name)

def printAttrs(name, obj):
    print(name)
    for key, val in obj.items():
        print("    %s:  %" % (key, val))

def findItem(name):
    if name in name:
        return name

def getKeys(HDFile,groupStr):
    k = HDFile[groupStr]
    l = list(k.keys())
    return 1

def readHDF(inputFile, debug=False):
    a = []
    with HDF.File(inputFile,'r+') as f:
        if debug: print(f.name)
        l = list(f.keys()) #extract keys in root dir of HDF file
        data = f["Data"]
        masks = f["Masks"]
        j = getKeys(f,"Masks")
        k = getKeys(f,"Data")
        wbc, oth, count = [], [], 0  # settings some initial values
        mT = f["data/"+k[count]][...,0]
        for p in k:
            if debug: print(count) # remove later
            m1 = f["Masks/"+j[count]][...] #Likewise for the mask data
            if debug: print(np.shape(m1))
            d1 = f["Data/"+[count]][...] #numpy style referencing to return nnumpy array of data
            if debug: print(np.shape(d1))
            if count == 0:
                if debug: print('wat') #to be removed
                wbc = d1[m1.astype(bool)]               #compile a dataset of wbc vectors
                oth = d1[np.invert(m1.astype(bool))]    #compile a dataset of non-wbc vectors
                tot = d1[mT.astype(bool)]               #compile a dataset of all vectors
                cls = m1.ravel()
            else:
                wbc = np.vstack((wbc, d1[m1.astype(bool)]))
                oth = np.vstack((oth, d1[np.invert(m1.astype(bool))]))  
                tot = np.vstack((tot, d1[mT.astype(bool)])) # appending vectors into band*(x*y) array
                cls = np.hstack((cls, m1.ravel()))
        count+=1
        if debug:
            print(np.shape(wbc))
            print(np.shape(oth))
            print(np.shape(tot))
            print(cls.shape)
        
    e = dcb_HDF5.saveClose(f)
    if(e): print("File " + inputFile + " saved Successfully!")


def buildHDF(inputFile, inputData, inputLabels, debug=False):
    """Creates a HDF file based on the data given, two possible datasets
    can be built, 25band set and 32 band set"""
    #Generate HDF5 file for input data stored as a list of datacube 
    HDFile = dcb_HDF5.openTable(inputFile)
    data_gp = dcb_HDF5.build25BandGroup(HDFile, "data")
    #determine the number of bands in the data, as well as band wavelenghts
    bands = inputData.shape()
    data_gp.attrs["RPArrowPrism"] = str(bands[2]) + ' Bands'
    label_gp = dcb_HDF5.build25BandGroup(HDFile, 'label')
    for p,  data in enumerate(inputData):  #with fix in create_bsq_dataset, now masks and data should be in same data struct
        if debug:
            if p == len(inputData)-1: print(p, inputData[p].Name, "LAST ITEM")
            if (inputData[p].Name == data.Name): print (inputData[p].Name)

        labelSet = dcb_HDF5.addDataset(label_gp,inputData[p].Name,inputData[p].MaskData,debug)
        dataSet = dcb_HDF5.addDataset(data_gp,inputData[p].Name,inputData[p].HSdata, debug)
        dataSet.attrs["units"] = "AU"
        labelSet.attrs["Parent_DCB"] = inputData[p].Name
    dcb_HDF5.saveClose(HDFile)  #add label to file under correct filename

###############################################################################


def main(argv):
    inputPath = ''
    outputfile = ''
    overwrite = False
    try:
        opts, args = getopt.getopt(argv,"hi:o",["iPath=","ofile="])
    except getopt.GetoptError:
        print('create_bsq_dataset.py -i <inputPath> -o <outputFile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Read_DCB_H5.py -i <inputPath> -o <outputFile>')
            sys.exit()
        elif opt in ("-i", "--ipath"):
            inputFile = arg
            #Add read HDF5 function here.

            #data = iterBSQ(inputPath)
            print('Input DIR = "', inputPath)
        elif opt in ("-o", "--ofile"):
            outputFile = arg
            print('Output File = "', outputfile)



if (__name__=="__main__"):
    # run this code when executed as main from terminal
    main(sys.argv[1:])

