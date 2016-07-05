from numpy import *
import matplotlib.pyplot as plt
import struct, os
from PIL import Image
#import imaging
import sys
sys.path
#from libtiff import TIFFfile
#import tifffile.c as tiff
import tifffile as TIFFfile
from scipy import io
import glob ## used to count files in a directory

sys.path
## =========================================================================================================
def imshow(image, ax=None, **kwargs):
    image = array(image)
    iscolor = (image.ndim == 3)

    if (ax == None):
        ax = plt.gca()
    def myformat_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if iscolor:
            (Nx,Ny,_) = image.shape
        else:
            (Nx,Ny) = image.shape

        if (col >= 0) and (col < Ny) and (row >= 0) and (row < Nx):
            if iscolor:
                z = image[row,col,:]
            else:
                z = image[row,col]

            if iscolor or (int(z) == z):
                if iscolor:
                    return('x=%i, y=%i, z=(%i,%i,%i)' % (row, col, z[0], z[1], z[2]))
                else:
                    return('x=%i, y=%i, z=%i' % (row, col, z))
            else:
                return('x=%i, y=%i, z=%1.4f' % (row, col, z))
        else:
           return('x=%i, y=%i'%(y, x))

    plt.imshow(image, **kwargs)
    ax.format_coord = myformat_coord
    return

## =========================================================================================================
def readbsq(filename, debug=False):
    filebytes = open(filename, "rb").read()
    output = struct.unpack('<25c', filebytes[0:25])
    #vv = output.join()
    vv = list(output)
    #in python 3.x you need to put a b'' instead of '' to signify you're reading bytes into a string/bytearray
    #putting a b in front of a string converts it into a byte array.  Why did the docs never mention this? beats me.
    version_string = b''.join(vv)
    #outputL = output.decode()

    #version_string = ''.join(outputL)



    if debug: print(version_string + '\n')

    if (version_string == b'RPSpectralCube v 0.1\x00\x00\x00\x00\x00'):
        version_number = 0
        firstbyte = 25
    elif (version_string == b'RPSpectralCube\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'):
        version_number = int32(struct.unpack('>I', filebytes[25:29])) #[0]
        firstbyte = 29
    else:
        raise ImportError('readbsq() cannot recognize that BSQ file version.')

    print('BSQ file version = ' + str(version_number))

    (Nx,Ny,Nw) = struct.unpack('>III', filebytes[firstbyte:firstbyte+12])
    firstbyte += 12
    if debug: print(Nx,Ny,Nw,firstbyte)

    ## From the Nw value, read in the set of wavelengths, each of type float64.
    lambdas = zeros(Nw)
    lambdas = struct.unpack('>' + str(Nw) + 'd', filebytes[firstbyte:firstbyte+(Nw*8)])
    firstbyte += Nw*8
    if debug: print('lambdas=', lambdas)

    if (version_number == 0):
        pass
    elif (version_number == 101):
        pixmin = struct.unpack('>d', filebytes[firstbyte:firstbyte+8])
        pixmax = struct.unpack('>d', filebytes[firstbyte+8:firstbyte+16])
        firstbyte = firstbyte + 16
        if debug: print('pixmin,pixmax = ', pixmin, pixmax)
    else:
        raise NotImplementedError('That version number (' + str(version_number) + ') is not implemented yet!')

    dcb = zeros((Nx,Ny,Nw))
    for w in arange(Nw):
        for y in arange(Ny):
            dcb[:,y,w] = struct.unpack('<'+str(Nx)+'f', filebytes[firstbyte:firstbyte+(Nx*4)])
            firstbyte += Nx*4

    return([dcb, lambdas])

## ============================================================================================
def write_16bitTiff(filename, image):
    img = image.copy()
    img = img[::-1,:]

    if img.dtype is not int16:
        if (amin(img) < 0): img = img - amin(img)
        img = float32(img) * (2.0**7 - 1.0) / amax(img)
        img = int8(img)

    ## The first method uses the "Image" library and converts the image to a string before converting to a PIL image.
    im = Image.fromarray(img)
    im.save(filename)

    ## The "Image" library doesn't allow compression. The "libtiff" library *does* allow compression:
    #from libtiff import TIFFimage
    #
    #tiff = TIFFimage(img, description='')
    #tiff.write_file(filename, compression='lzw')
    #del tiff

    return

## ============================================================================================
def write_dcb_tiffs(fileroot, dcb, lambdas = None):
    (Nx,Ny,Nw) = dcb.shape

    if (lambdas == None):
        lambdas = []
        for w in arange(Nw):
            lambdas.append('%02i' % w)

    for w in arange(Nw):
        wavestr = ('%03i' % int(lambdas[w]))
        filename = fileroot + "-" + wavestr + '.tif'
        print('Writing filename "' + filename + '"')
        img = dcb[:,:,w]
        write_16bitTiff(filename, img)

    return

## ============================================================================================
def readtiff(filename):
    assert os.path.exists(filename), 'File "' + filename + '" does not exist.'
    tif = TIFFfile(filename)
    img = tif.asarray()
    tif.close()
    img = flipud(img)
    return(img)


## ============================================================================================
#main section of convertBsqMulti.py

if __name__ == "__main__":
    pathName = r'/media/crob/HyperSpec/-_Last_Minute_Data_-'  #F:\-_Research Data_-\Blood 9_3_2015\Slide 1  /media/crob/USB30FD/HyperSpec_Data/All bsq static
    os.chdir(pathName)
    #objNames = glob.glob1(pathName,"*.bsq")
    #bsqCount = len(objNames)
    ##  bsqCount = len(glob.glob1(pathName,"*.bsq")) Need both number of files as well as filenames

    ## This loop repeats for all subfolders
    all_subDirs = [d for d in os.listdir('.') if os.path.isdir(d)]

    for dirs in all_subDirs:
        sDir = os.path.join(pathName, dirs)
        objNames = glob.glob1(sDir,"*.bsq")
        bsqCount = len(objNames)

        ## insert for loop here
        for i in range(0,bsqCount):

            dir = os.path.dirname(os.path.join(sDir,objNames[i]))

            if not os.path.exists(dir+'/TIF/'):
                os.mkdir(dir+'/TIF/')
            outfile = dir + '/TIF/' + os.path.basename(os.path.join(sDir,objNames[i]))[:-4]
            (dcb, lambdas) = readbsq(os.path.join(sDir,objNames[i]))

            print('dcb.shape=', dcb.shape)
            print('dcb.dtype=', dcb.dtype)
            print('lambdas=', lambdas)

        ## Convert type double pixel data to type uint16 for exporting to TIFF.
            minval = amin(dcb)
            maxval = amax(dcb)
            print('original dcb: minval=%f, maxval=%f' % (minval, maxval))

            new_dcb = int16((dcb - minval) * (2**15 - 1) / float64(maxval - minval))
            print('new_dcb.dtype=', new_dcb.dtype)
            print('rescaled dcb: minval=%i, maxval=%i' % (amin(new_dcb), amax(new_dcb)))

            img = dcb[:, :, 0]
            im = Image.fromarray(img)
            #im.show()
            #imshow(img)

        ## Write the result as a 16-bit TIFF file. Note that very few TIFF viewers support
        ## 16-bit pixels! ImageJ is a free and widely available one, though.
            write_dcb_tiffs(outfile, dcb, lambdas)

        ## Finally, write out the BSQ file as a Matlab-style .mat file.
            #io.savemat(outfile, mdict={'dcb':dcb})

            #plt.show()


