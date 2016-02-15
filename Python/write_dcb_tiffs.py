#!/usr/bin/python
import numpy as np
from PIL import Image


def write_dcb_tiffs(fileroot, dcb, lambdas = None):
    (Nx,Ny,Nw) = dcb.shape

    if (lambdas == None):
        lambdas = []
        for w in np.arange(Nw):
            lambdas.append('%02i' % w)

    for w in np.arange(Nw):
        wavestr = ('%03i' % int(lambdas[w]))
        filename = fileroot + "-" + wavestr + '.tif'
        print('Writing Image: "' + filename + '"')
        img = dcb[:,:,w]
        write_16bitTif(filename, img)

    return


def write_16bitTif(filename, image):
    img = image.copy()
    img = img[::-1,:]

    if img.dtype is not np.int16:
        if (np.amin(img) < 0): img = img - np.amin(img)
        img = np.float32(img) * (2.0**15 - 1.0) / np.amax(img)
        img = np.int16(img)

    im = Image.fromarray(img)
    im.save(filename)
    
    return
    

if __main__=="__main__":
	# there shouldn't be anything here really maybe a test?
	print('write_dcb_tiffs.py')
