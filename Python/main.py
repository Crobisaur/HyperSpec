__author__ = 'Christopher Robison'

#packages required numpy scipy matplotlib pyside opencv3.0 pillow libtiff?

#read in bsq files and mask files.  Find a way to display them to make sure
#masks are aligned properly.

import sys

#import dataCube

import matplotlib
matplotlib.use('Qt4Agg')

import convertBsqMulti as BSQ
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import scipy
#import matplotlib.pyplot as plt
import struct, os
from PIL import Image
from PIL import ImageOps
#import tifffile as TIFFfile
from scipy import io
import glob
import skimage.io
from dataCube import *
import dcb_HDF5

#use from when importing objects from modules (i.e. a data structure object from a data structure module)

def readBsqToObj(path,mPath):
    dCube = dataCube()
    (dCube.HSData, dCube.Lambdas) = BSQ.readbsq(dir)
    #add in line to read in mask image
    return dCube

def readMaskToObj(dir,maskList):
    dCube = dataCube()
    dCube.MaskData = Image.open(r'MaskPath')
    return dCube

def showImage(im, show = False):
    #Will not show image by default, only need to show for debug purposes.
    if(show):
        im = Image.fromarray(im)
        im.show()

display_min = 1000
display_max = 10000
def display(image, display_min, display_max):
    image = np.array(image, copy=True)
    image.clip(display_min, display_max, out=image)
    image -= display_min
    image //= (display_max - display_min + 1)/ 256.
    return image.astype(np.uint8)

def lut_display(image, display_min, display_max):
    lut = np.arange(2**16, dtype='uint16')
    lut = display(lut, display_min, display_max)
    return np.take(lut, image)


def readBsqsinDir(path,dataName,maskName):
    #loads all bsq files and mask files into memory for processing

    sDir = os.path.join(pathName, dirs)
    objNames = glob.glob1(sDir,"*.bsq")
    bsqCount = len(objNames)

    #loop over all files in dir
    for i in range(0,bsqCount):
        dir = os.path.dirname(os.path.join(sDir,objNames[i]))


def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr



# Work Path(s) r'F:\-_Research Data_-\Blood 9_3_2015\Slide 1'
# Home Path(s) r'D:\-_RESEARCH DATA_-\DataCubes' r'D:\-_RESEARCH DATA_-\DataMasks'
wkPath = r'F:\-_Thesis Data_-\DataCubes'
hmPath = r'D:\-_RESEARCH DATA_-\DataCubes'

pathName = hmPath
maskDirName = r'masks'
bsqDirName = r'bsq'
os.chdir(pathName)

if not os.path.exists(pathName+'/processed_masks/'):
    os.mkdir(pathName+'/processed_masks/')
#objNames = glob.glob1(pathName,"*.bsq")
#bsqCount = len(objNames)
##  bsqCount = len(glob.glob1(pathName,"*.bsq")) Need both number of files as well as filenames

## This loop repeats for all subfolders
all_subDirs = [d for d in os.listdir('.') if os.path.isdir(d)]
dataList = []
maskList = []
for dirs in all_subDirs:
    sDir = os.path.join(pathName, dirs)
    objNames = glob.glob1(sDir,"*.bsq")
    bsqCount = len(objNames)
    maskNames = glob.glob1(sDir,"*.tif")
    maskCount = len(maskNames)
    print(sDir)

    #This loop runs when looking in Mask Directory
    if dirs == 'masks':
        for i in range(0,maskCount):
            # dir = os.path.dirname(os.path.join(sDir,objNames[]))
            mFile = os.path.join(sDir, maskNames[i])
            mDir = os.path.dirname(mFile)
            print(mFile[:-4] + '\r\n' + mDir)
            im = np.rot90(TIFFfile.imread(mFile), 2)
            im = im[:,:,0]
            im[im < np.max(im)] = 0
            im[im == np.max(im)] = 1
            scipy.misc.imsave(pathName + '\\processed_masks\\' + maskNames[i][2:-8] + '.png', im)
            # im = skimage.io.imread(mFile, plugin='tifffile')
            A = dataCube(maskNames[i][2:-8], None, None, im)
            print(A.MaskData.shape)
            maskList.append(A)

    ## insert for loop here  (add if statement)
    #This loop runs when looking in BSQ Directory
    if dirs == 'bsq':
        for i in range(0,bsqCount):
            dir = os.path.dirname(os.path.join(sDir,objNames[i]))
            print(str(maskCount))
            if not os.path.exists(dir+'/TIF/'):
                os.mkdir(dir+'/TIF/')
            #outfile = dir + '/TIF/' + os.path.basename(os.path.join(sDir,objNames[i]))[:-4]
            (dcb, lambdas) = BSQ.readbsq(os.path.join(sDir,objNames[i]))

            #print('dcb.shape=', dcb.shape)
            #print('dcb.dtype=', dcb.dtype)
            #print('lambdas=', lambdas)

        # Convert type double pixel data to type uint16 for exporting to TIFF.
            minval = np.amin(dcb)
            maxval = np.amax(dcb)
            #print('original dcb: minval=%f, maxval=%f' % (minval, maxval))

            new_dcb = np.int16((dcb - minval) * (2**15 - 1) / np.float64(maxval - minval))
            #print('new_dcb.dtype=', new_dcb.dtype)
            #print('rescaled dcb: minval=%i, maxval=%i' % (amin(new_dcb), amax(new_dcb)))

            img = new_dcb[:, :, 0]
            im = Image.fromarray(img)
            ll = np.array(lambdas)
            print(str(ll))
            lambdas = ["%.2f" % elem for elem in lambdas] # format lambdas to 2 decimals (for sanity's sake)
            B = dataCube(objNames[i][:-4], new_dcb, lambdas,None)
            dataList.append(B)
            #im.show()
            #imshow(img)

    # Write the result as a 16-bit TIFF file. Note that very few TIFF viewers support
    # 16-bit pixels! ImageJ is a free and widely available one, though.
        #BSQ.write_dcb_tiffs(outfile, dcb, lambdas)

    # Finally, write out the BSQ file as a Matlab-style .mat file.
        #io.savemat(outfile, mdict={'dcb':dcb})

        #plt.show()


def lookahead(iterable):
    it = iter(iterable)
    last = next(it) #it.next() in python 2
    for val in it:
        yield last, False
        last = val
        yield last, True


#all_dataCubes = [d for d in dataList]
first = True
hsList = []
HDFile = dcb_HDF5.createTable("HS_thesis_data_v3-1.h5")
d_group = dcb_HDF5.build25bandGroup(HDFile, "Data")
d_group.attrs["RPArrowPrism"] = '25 Bands'
lSet = dcb_HDF5.addDataset(HDFile,"Prism25_Band_Lambdas",ll)
lSet.attrs["units"] = "nanometers"
m_group = dcb_HDF5.build25bandGroup(HDFile, "Masks")
m_group.attrs["MaskFmt"]= "logical"

for p, data in enumerate(dataList):
    print(maskList[p].Name, data.Name)
    #dcbGroup = dcb_HDF5.build25bandGroup(d_group,dataList[p].Name)
    if maskList[p].Name == data.Name:
        #print('MaskData size ', np.shape(maskList[p]))
        #print(data.Lambdas)
        dataList[p].MaskData = maskList[p].MaskData
        maskSet = dcb_HDF5.addDataset(m_group,maskList[p].Name,maskList[p].MaskData)
        maskSet.attrs["Parent_DCB"] = dataList[p].Name
        #data.setMaskData(maskList[p].MaskData)
        #hsList.append(data)
    if p == len(dataList)-1:
        print(p, data.Name, "LAST ITEM")  #only happens when last iteration of for loop runs
    print(data.Name + str(p))
    tTable = dcb_HDF5.addDataset(d_group,data.Name,dataList[p].HSdata)
    tTable.attrs["units"] = "AU"
    HDFile.flush()

    #tTable = thesis_TABLE.addTable2Group(d_group,HDFile,data)
    #thesis_TABLE.add25Vector2DataCube(tTable,data)


dcb_HDF5.saveClose(HDFile)
ddd = dataList[1].HSdata
#print(np.shape(ddd)[2])

im2 = dataList[1].MaskData

rd = ddd[im2.astype(bool)]  #returns masked by im2
id = ddd[np.invert(im2.astype(bool))]
#print([np.shape(rd), np.count_nonzero(im2)])
#print([np.shape(ddd), np.shape(im2)])
samp = ddd[:,:,0]
#out = Image.fromarray(im2[:,:,0])
maskplot = plt.imshow(np.float32(im2))
dataplot = plt.imshow(np.float32(samp)) #imshow only works with 32bit grayscale

scipy.misc.imsave(pathName+'Mask.png', im2)
scipy.misc.imsave(pathName+'Data.png', samp)
scipy.misc.imsave(pathName+'WBCData.png', np.rot90(rd,3))
scipy.misc.imsave(pathName+'ElseData.png', np.rot90(id,3))
#print(rd[1])


plt.show(maskplot)
#maskplot.savefig('Mask.png')

plt.show(dataplot)
#dataplot.savefig('Data.png')

#samp = Image.fromarray(ddd[:,:,0],'F')

#samp = ddd[:,:,0]
#ImageOps.autocontrast(samp)
np.savetxt('TestImg2.csv', ddd[:,:,0],delimiter=',',fmt='%5.0d')
np.savetxt('TestOut.csv', np.rot90(rd,3), delimiter=',',fmt='%5.0d')
#np.savetxt('TestMsk.csv', im2[:,:,0],delimiter=',', fmt='%5.0d')
#lut_display(samp, display_min, display_max)

#HDFile = thesis_TABLE.createH5File("HS_thesis_data.h5", "TEST DATA")
#d_group = thesis_TABLE.build25bandGroup(HDFile)
#tTable = thesis_TABLE.addTable2Group(d_group,HDFile,dataList[1])
#thesis_TABLE.add25Vector2DataCube(tTable,dataList[1])
