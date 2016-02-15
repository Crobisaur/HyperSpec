#!/usr/bin/python
from dataCube import dataCube

import numpy as np
import glob, os
from scipy import misc
from PIL import Image



# this program will loop through hsi data directory containing /data and /mask folders.  It will match corresponding masks with data and import all files into a LMDB for caffe NN
def findBSQ(path, cubeName, debug=False):
    """This function will return a datacube object containing datacube data of one cube in the given directory"""
    (dcb, lambdas) = readbsq(os.path.join(path, cubeName))
    minval = np.amin(dcb)
    maxval = np.amax(dcb)
    # there is some compression done here but only needed for tifs so nah
    lambdas = ["%.2f" % elem for elem in lambdas] # format lambdas to 2 decimal places for neatness
    if debug: print(str(np.array(lambdas))
    B = dataCube(cubeName[:-4], dcb, lambdas, None)
    if debug: print(B.HSData.shape)

    return B


def findMask(path, maskName, debug=False):
    """This function will return a datacube object containing mask data of one cube in the given directory"""
    mask_file = os.path.join(path, maskName)
    if debug: print(mask_file[:-4] + '\r\n' + path)
    mask_im = misc.imread(mask_file)
    if debug: print(np.shape(mask_im))
    # Normalization
    # mask_im[mask_im < np.max(mask_im)] = 0
    # mask_im[mask_im == np.max(mask_im)] = 1
    # Save binary mask as image
    # *** Add fucntionality to convert mask into separate binary masks based on rgb values***
    A = dataCube(maskName[2:-8], None, None, mask_im)
    if debug: print(A.MaskData.shape)
    # return datacube object containing mask data
    return A


def iterBSQ(path, debug=False):
    """This function finds all sub directories in a given dir and iterates through all of
    them to find datacube files and mask files stored as .bsq and .png respectively."""
    os.chdir(path)
    all_subDirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    dataList = []
    maskList = []
    for dirs in all_subDirs:
        subDir = os.path.join(path, dirs)
        objNames = glob.glob1(subDir,"*.bsq")
        bsq_count = len(objNames)
        maskNames = glob.glob1(subDir,"*.png")  # this was "*.tif" but tif files are a pain
        mask_count = len(maskNames)
        if debug: print(subDir)

        #This loop runs when looking in Mask Dir
        # This loop assumes data is tif, maybe make a function that loops for specific file type?
        if dirs == 'masks':
            for i in range(0,mask_count):
                M = findMask(subDir, maskNames[i])
                maskList.append(M)

        if dirs == 'bsq':
            for i in range(0,bsq_count):
                D = findBSQ(subDir, objNames[i])
                dataList.append(D)


                #mask_file = os.path.join(subDir, maskNames[mask])
                #mask_dir = os.path.dirname(mask_file)
                #aif debug: print(mask_file[:-4] + '\r\n' + mask_dir)


def main(argv):
    inputPath = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o",["iPath=","ofile="])
    except getopt.GetoptError:
        print('create_bsq_dataset.py -i <inputPath> -o <outputFile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('create_bsq_dataset.py -i <inputPath> -o <outputFile>')
            sys.exit()
        elif opt in ("-i", "--ipath"):
            inputFile = arg
            data = iterBSQ(inputPath)
            print('Input DIR = "', inputPath)
        elif opt in ("-o", "--ofile"):
            outputFile = arg
            print('Output File = "', outputfile)


if (__name__=="__main__"):
    # run this code when executed as main from terminal
    main(sys.argv[1:])