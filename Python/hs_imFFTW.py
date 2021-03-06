__author__ = 'Christo Robison'

#This program preprocesses the data by removing artifacts caused by the snapshot HSI system

from spectral import *
import numpy as np
from PyQt4 import QtGui
import pyqtgraph as pg
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
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
import HyperSpecGui

#x = np.random.normal(size=1000)
#y = np.random.normal(size=1000)

#pg.plot(x, y, symbol='o')
#pg.QtGui.QApplication.exec_()

show_full_out = False
if show_full_out: np.set_printoptions(threshold=np.nan)


def getData(filename=None, dat_idx=None, lab_idx=None):
    if filename is None: filename = 'D:\-_Hyper_Spec_-\HYPER_SPEC_TEST.h5'
    f = h5py.File(filename, 'r')
    if dat_idx is None:
        dcb = f['data'][:] #Extract normalized data for svm b/c intensity sensitive
    else:
        dcb = f['data'][:, :, dat_idx:dat_idx+len(f['bands'])]

    if lab_idx is None:
        labels = f['labels'][:]
        classLabels = f['classLabels'][:]
    else:
        labels = f['labels'][:, :, lab_idx]
        classLabels = f['classLabels'][:, :, lab_idx]

    bands = f['bands'][:]
    #classLabels = f['classLabels'][:]
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


def hsfft(dcb):
    '''Takes in a datacube (dcb) and returns n-dim FFT of datacube'''
    temp_dcb = pyfftw.n_byte_align(dcb, 16, dtype='complex128')
    dcb_fft = pyfftw.interfaces.numpy_fft.fftn(temp_dcb)
    return dcb_fft


def genMask(h=443, w=313, offset=41, debug=False):
    '''Generates a fft mask to eliminate HS system artifacts from image'''
    mask = np.ones((h, w), dtype='float32')
    mask[:, (w/2):((w+2//2)//2)] = 0
    mask[((h/2)-offset):((h/2)+offset), :] = 1
    if debug: print(mask)
    return mask


def applyMask(dcb_fft, mask=genMask, debug=False):
    '''Applies mask to spatial information of each band in datacube.'''
    masked_fft = np.multiply(np.rollaxis(dcb_fft, 2, 0,), ft.ifftshift(mask))
    if debug: print("Shape of FFT:" + str(np.shape(masked_fft)))
    return masked_fft


def hsifft(dcb_fft, debug=False):
    '''Performs inverse fourier transform on datacube'''
    dcb_out = pyfftw.interfaces.numpy_fft.ifftn(dcb_fft)
    if debug: print("Shape of iFFT:" + str(np.shape(dcb_out)))
    return dcb_out


def dcbFilter(dcb, h=443, w=313, offset=41, debug=False):
    '''A helper function to simplify masking process and reduce confusion'''
    masked_fft = applyMask(hsfft(dcb), genMask(h, w, offset, debug))
    dcb_out = hsifft(masked_fft, debug)
    return dcb_out


def dispDCB(dcb):
    dcb = np.swapaxes(dcb, 2, 0)


if __name__ == '__main__':
    # initialize Qt, needed once per application
    app = QtGui.QApplication([])
    # Define widget to hold outputs
    MainWindow = QtGui.QMainWindow()
    MainWindow.setWindowTitle('HyperSpec: Test')
    ui = HyperSpecGui.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()


    #widg = QtGui.QWidget()
    #ploot = pg.PlotWidget()
    #layout = QtGui.QGridLayout()
    #widg.setLayout(layout)
    # add widgets to layout
    #layout.addWidget(ploot, 0, 1, 3, 1)

    #widg.show()
    #app.exec_()


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


    ##### segment off each section, simplify repetitive code####

    ##### Pull out a data cube and display it in a pyqtgraph object#####


    img = trainData['dcb'][:, :, 343:370]
    img1 = np.swapaxes(img, 2, 0)

    ui.ImageView1.setImage(img1)


    # (m, c) = kmeans(img, 6, 300)
    img_file = open('fftMask_new.png', 'wb')
    img_w = png.Writer(313, 443, greyscale=True, bitdepth=16)
    img_to_write = exposure.rescale_intensity(img[:, :, 3], out_range='float')
    img_to_write = img_as_uint(img_to_write)
    img_w.write(img_file, img_to_write)
    img_file.close()

    # png.from_array(img[:, :, 3]).save("fftMask.png")
    #imv1 = pg.ImageView()
    #imv.show()
    #imv.setImage(img[:,:,3])
    #pre_img = imshow(img[:, :, 3])
    ui.ImageView2.setImage(img[:, :, 3])

    #pg.QtGui.QApplication.exec_()
    # plt.savefig('fft3_pre')

    mask = np.ones((443, 313), dtype='float32')
    mask[:, 155:157] = 0  #10:430
    mask[180:262, :] = 1
    print(mask)


    #mask_img = imshow(mask)
    ui.ImageView4.setImage(mask)
    #pg.QtGui.QApplication.exec_()
    # imSave(mask_img,'Mask_img.png')
    # imSave(mask, "Mask_img.png", out_range=np.float32)
    # plt.savefig('fftMask')


    #for i in range(np.shape(img)[2]):

    #    print(i)

    temp_img = pyfftw.n_byte_align(img, 16, dtype='complex128')
    img_fft = pyfftw.interfaces.numpy_fft.fftn(temp_img)
    print(img_fft.dtype)
    print(img_fft.shape)
    #if projecting mask along data, need to make sure last two dimensions match  ex 4Lx3Wx10H * 3Wx10H = 4Lx3Wx10H

    mask_ifft = np.multiply(np.rollaxis(img_fft,2,0), ft.ifftshift(mask))
    print("Shape of FFT: " + str(np.shape(mask_ifft)))
    #mask_ifft = np.rollaxis(mask_ifft,0,3)
    print("Shape of FFT: " + str(np.shape(mask_ifft)))

    img_FFTout = np.log2(pyfftw.interfaces.numpy_fft.ifftn(mask_ifft))
    print("Shape of FFT_Out: " + str(np.shape(img_FFTout)))
    # imSave(img_FFTout, "OUTPUT_FFT.png")
    img_p = pyfftw.interfaces.numpy_fft.ifftn(mask_ifft)
    ui.ImageView3.setImage(img_p)


    #Start implementing parts from final_tests.py
    (m, c) = kmeans(img_p.real, 6, 300)
    mw = pg.widgets.MatplotlibWidget.MatplotlibWidget()
    subplot = mw.getFigure().add_subplot(111)
    for i in range(c.shape[0]):
        subplot.plot(c[i])
    mw.draw()




    #mask_out = imshow(pyfftw.interfaces.numpy_fft.ifftn(mask_ifft))
    #pg.QtGui.QApplication.exec_()
    # mask_hist = imshow(np.histogram(img_p.real))
    #pg.image(img_p)
    #pg.QtGui.QApplication.exec_()

    #plt.clf()
    #plt.hist(img_p.real, bins='auto')
    #plt.title("Histogram of image")
    #plt.show()
    
    
    #img_o = img_p.real
    #im2png(img_o, "output_image.png")
    #imgg = imshow(img_o)
    #plt.savefig('mask_FFT')
    #fft_real = mask_ifft.real
    #fft_imag = mask_ifft.imag
    #print(fft_real.dtype)
    #print(fft_imag)


    #png.from_array(np.array(fft_real, dtype=np.uint16),'L').save("Mask_FFT_real.png")
    #imSave(fft_real, "Mask_FFT_real.png", range='float')
    #imSave(fft_imag, "Mask_FFT_imag.png", range='float')

    #imSave(abs(pyfftw.interfaces.numpy_fft.ifftn(mask_ifft)), "Mask_fft_1.png", out_range=np.uint8)
    #png.from_array(pyfftw.interfaces.numpy_fft.ifftn(mask_ifft)).save("mask_FFT.png")
    #real_fft = np.abs(np.log2(img_fft))
    #out_img = imshow(ft.fftshift(real_fft))
    #png.from_array(ft.fftshift(real_fft)).save("fft3.png")
    #plt.savefig('fft3')

    sys.exit(app.exec_())