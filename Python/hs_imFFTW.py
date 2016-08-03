__author__ = 'Christo Robison'

#This program preprocesses the data by removing artifacts caused by the snapshot HSI system

from spectral import *
import numpy as np
from scipy import fftpack as ft
import h5py
import matplotlib
#matplotlib.use('qt4')
#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import time
from skimage import io, exposure, img_as_uint, img_as_float
import png
#io.use_plugin('freeimage')
import pyfftw

show_full_out = False
if show_full_out: np.set_printoptions(threshold=np.nan)

def getData(filename=None):
    if filename is None: filename = 'D:\-_Hyper_Spec_-\HYPER_SPEC_TEST.h5'
    f = h5py.File(filename, 'r')
    dcb = f['data'][:] #Extract normalized data for svm b/c intensity sensitive
    labels = f['labels'][:]
    bands = f['bands'][:]
    classLabels = f['classLabels'][:]
    out = {'dcb': dcb, 'labels': labels, 'lambdas': bands, 'classLabels': classLabels}
    f.close()
    return out

    
def im2png(img, filename):
    f = open(filename, 'wb')
    w = png.Writer(313, 443, greyscale=True, bitdepth=8)
    if img.dtype is not np.int8:
        if (np.amin(img) < 0): img = img - np.amin(img)
        img = np.float32(img) * (2.0**7 - 1.0)/ np.amax(img)
        img = np.int8(img)
    #img_w = exposure.rescale_intensity(img, out_range='int8')
    w.write(f, img)
    f.close()
        

def imSave(img, filename, range='float'):
    f = open(filename,'wb')
    w = png.Writer(313, 443, greyscale=True, bitdepth=8)
    img_w = exposure.rescale_intensity(img,out_range=range)
    img_w = img_as_uint(img_w)
    w.write(f, img_w)
    f.close()


def shapeData(data, labels, numExamples, numBands, altDims = None):
    '''Takes input data matrix and reshapes it into HW,D format
    i.e. endmembers and their appropriate class labels'''
    if altDims is None: altDims = [443, 313, numBands, numExamples]
    temp = np.reshape(data, altDims, 'f')
    dataR = np.reshape(temp,[-1, numBands])
    labelL = np.reshape(labels, [-1,1])
    out = {'data': dataR, 'label': labelL}
    return out


if __name__ == '__main__':
    trainData = getData(filename='D:\-_Hyper_Spec_-\HYPER_SPEC_TEST_RED.h5')
    testData = getData(filename='D:\-_Hyper_Spec_-\HYPER_SPEC_TEST_RED.h5')
    print(np.shape(trainData['dcb']))
    debug = False
    if debug is True:
        for i in range(np.shape(trainData['dcb'])[2]):
            im = exposure.rescale_intensity(trainData['dcb'][:, :, i], out_range='float')
            im = img_as_uint(im)
            io.imsave((r'HYPER_SPEC_TEST\band_image_' + str(i) + '.png'), im)

            # pf = open(('band_image_' + str(i) + '.png'), 'wb')
            # w = png.Writer(width=313, height=443, bitdepth=16, greyscale=True)
            # w.write(pf, np.reshape(testData['dcb'], (-1, 443 * 372)))
            # pf.close()

    ### Unsupervised Classification
    # img = trainData['dcb'][:,:,1625:1651]
    # (m, c) = kmeans(img, 6, 300)
    img = trainData['dcb'][:, :, 343:370]
    #(m, c) = kmeans(img, 6, 300)
    img_file = open('fftMask_new.png', 'wb')
    img_w = png.Writer(313, 443, greyscale=True, bitdepth=16)
    img_to_write = exposure.rescale_intensity(img[:, :, 3], out_range='float')
    img_to_write = img_as_uint(img_to_write)
    img_w.write(img_file, img_to_write)
    img_file.close()

    #png.from_array(img[:, :, 3]).save("fftMask.png")
    pre_img = imshow(img[:, :, 3])
    #plt.savefig('fft3_pre')

    mask = np.ones((443,313), dtype='float32')
    mask[20:420, 155:157] = 0
    mask[160:282, :] = 1
    print(mask)
    mask_img = imshow(mask)
    #imSave(mask_img,'Mask_img.png')
    #imSave(mask, "Mask_img.png", out_range=np.float32)
    #plt.savefig('fftMask')



    temp_img = pyfftw.n_byte_align(img[:, :, 3], 16, dtype='complex128')
    img_fft = pyfftw.interfaces.numpy_fft.fftn(temp_img)
    print(img_fft.dtype)
    print(img_fft.shape)
    mask_ifft = np.multiply(img_fft, ft.ifftshift(mask))
    img_FFTout = np.log2(pyfftw.interfaces.numpy_fft.ifftn(mask_ifft))
    #imSave(img_FFTout, "OUTPUT_FFT.png")
    img_p = pyfftw.interfaces.numpy_fft.ifftn(mask_ifft)
    mask_out = imshow(pyfftw.interfaces.numpy_fft.ifftn(mask_ifft))
    #mask_hist = imshow(np.histogram(img_p.real))

    plt.clf()
    plt.hist(img_p.real, bins='auto')
    plt.title("Histogram of image")
    plt.show()
    
    
    img_o = img_p.real
    im2png(img_o, "output_image.png")
    imgg = imshow(img_o)
    #plt.savefig('mask_FFT')
    fft_real = mask_ifft.real
    fft_imag = mask_ifft.imag
    print(fft_real.dtype)
    print(fft_imag)


    #png.from_array(np.array(fft_real, dtype=np.uint16),'L').save("Mask_FFT_real.png")
    imSave(fft_real, "Mask_FFT_real.png", range='float')
    imSave(fft_imag, "Mask_FFT_imag.png", range='float')

    imSave(abs(pyfftw.interfaces.numpy_fft.ifftn(mask_ifft)), "Mask_fft_1.png", out_range=np.uint8)
    #png.from_array(pyfftw.interfaces.numpy_fft.ifftn(mask_ifft)).save("mask_FFT.png")
    real_fft = np.abs(np.log2(img_fft))
    out_img = imshow(ft.fftshift(real_fft))
    png.from_array(ft.fftshift(real_fft)).save("fft3.png")
    #plt.savefig('fft3')
