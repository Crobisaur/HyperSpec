__author__ = 'Christo Robison'

from PyQt4 import QtGui, QtCore
import sys
import HyperSpecGui
import numpy as np
from scipy import ndimage
from scipy import fftpack as ft
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

import time
import pyqtgraph
import hs_imFFTW as hs
from spectral import *
spectral.settings.WX_GL_DEPTH_SIZE = 16


'''
background = 0
red = 1 WBC
green = 2 RBC
pink = 3 nuclear material
yellow = 4 ignore'''

class HyperSpecApp(QtGui.QMainWindow, HyperSpecGui.Ui_MainWindow):
    def __init__(self, parent=None):
        pyqtgraph.setConfigOption('background', 'k')  #before loading widget
        super(HyperSpecApp, self).__init__(parent)
        self.setupUi(self)
        self.updateBtn.clicked.connect(self.update)

        #self.view = pyqtgraph.ViewBox(self.gView0)
        #self.scene = pyqtgraph.ImageItem(self.view)
        STEPS = np.array([0.0, 0.2, 0.6, 1.0])
        CLRS = ['k','r','y','w']
        clrmp = pyqtgraph.ColorMap(STEPS, np.array([pyqtgraph.colorTuple(pyqtgraph.Color(c)) for c in CLRS]))
        data = np.random.normal(size=(100, 200, 200))

        #imv = pyqtgraph.image(data)
        self.ImageView2.setImage(data)

        #self.ImageView2.ui.histogram.gradient.setColorMap(clrmp)






        #self.img_array = np.zeros((1000, CHUNKSZ / 2 + 1))

        # bipolar colormap

        #pos = np.array([0., 1., 0.5, 0.25, 0.75])
        #color = np.array([[0, 255, 255, 255], [255, 255, 0, 255], [0, 0, 0, 255], (0, 0, 255, 255), (255, 0, 0, 255)],
#                         dtype=np.ubyte)
        #cmap = pyqtgraph.ColorMap(pos, color)
        #lut = cmap.getLookupTable(0.0, 1.0, 256)
        #self.img.setLookupTable(lut)


        self.pltView1.plotItem.showGrid(True, True, 0.7)
        self.pltView2.plotItem.showGrid(True, True, 0.7)

    def update(self):
        t1 = time.clock()
        points = 100
        # Temp demo code
        #X = np.arange(points)
        #Y = np.sin(np.arange(points) / points * 3 * np.pi + time.time())
        #C = pyqtgraph.hsvColor(time.time() / 5 % 1, alpha=.5)
        #pen = pyqtgraph.mkPen(color=C, width=10)
        #self.pltView0.plot(X, Y, pen=pen, clear=True)
        print("update took %.02f ms" % ((time.clock() - t1) * 1000))
        #if self.chkMore.isChecked():
            #QtCore.QTimer.singleShot(1, self.update)  # QUICKLY repeat

    def done(self):
        QtGui.QMessageBox.information(self, "Done!", "Done ")


def runKmeans(view, inputImg, clusters=6, iters=300):
    (m, c) = kmeans(inputImg, clusters, iters)
    #mw = pyqtgraph.widgets.MatplotlibWidget.MatplotlibWidget()
    #subplot = form.pltView0.add_subplot(111)
    #form.pltView0.
    for i in range(c.shape[0]):
        view.plot(c[i])
    #form.pltView0.plot(subplot)
    return (m, c)

def adjustLabels(dcb, bkgnd=0.0, unknown=4.0):
    dcb[dcb==0.0] = 5.0
    dcb[dcb==4.0] = 0.0
    dcb[dcb==5.0] = 4.0
    return dcb

def runSpectral(dcb, gt, title = 'dcb'):
    (classes, gmlc, clmap) = runGauss(dcb, gt)
    (gtresults, gtErrors) = genResults(clmap, gt)
    displayPlots(clmap, gt, gtresults, gtErrors, (title+" Gaussian Classifer"))
    return (gtresults, gtErrors)

def runPCA(dcb, gt, title = 'dcb'):
    pc = principal_components(dcb)
    pc_0999 = pc.reduce(fraction=0.999)
    img_pc = pc_0999.transform(dcb)
    (classes, gmlc, clmap) = runGauss(img_pc, gt)
    (gtresults, gtErrors) = genResults(clmap, gt)
    displayPlots(clmap, gt, gtresults, gtErrors, (title+" PCA Gaussian Classifer"))
    return (gtresults, gtErrors, pc)

def genResults(clmap, gt):
    gtresults = clmap * (gt!=0)
    gtErrors = gtresults * (gtresults !=gt)
    return (gtresults, gtErrors)

def runGauss(dcb, gt):
    classes = create_training_classes(dcb, gt)
    gmlc = GaussianClassifier(classes, min_samples=200)
    clmap = gmlc.classify_image(dcb)
    return (classes, gmlc, clmap)

def displayPlots(clmap, gt, gtresults = None, gtErrors = None, title = 'classifier'):
    if (gtresults is None and gtErrors is None):
        (gtresults, gtErrors) = genResults(clmap, gt)
    v0 = imshow(classes=clmap, title=(title+" results"))
    pylab.savefig((title+" results.png"), bbox_inches='tight')
    v1 = imshow(classes = gtresults, title=(title+" gt Results"))
    pylab.savefig((title + " gt Results.png"), bbox_inches='tight')
    v2 = imshow(classes = gtErrors, title=(title+" Error"))
    pylab.savefig((title + " Error.png"), bbox_inches='tight')

def cleanResults(inputImg, cls_iter=1, open_iter=1):
    open = ndimage.binary_opening(inputImg, iterations=open_iter)
    close = ndimage.binary_opening(inputImg, iterations=cls_iter)
    return (open, close)


def combineLabels(rbc, wbc, nuc, bkgd):
    out = np.zeros(np.shape(rbc), dtype=np.float64)
    out[bkgd == 1] = 4.0
    out[rbc == 1] = 2.0
    out[wbc == 1] = 1.0
    out[nuc == 1] = 3.0
    out[out == 0] = 0.0

    print(out)
    return out

def create_rgb(classimg, colormap=None):
    if colormap is None:
        colormap = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [255, 0, 255], [255, 255, 0]], dtype=np.ubyte)
    h,w = np.shape(classimg)
    out = np.zeros([h, w, 3], dtype=np.uint8)
    out[classimg == 0.0] = colormap[0]
    out[classimg == 1.0] = colormap[1]
    out[classimg == 2.0] = colormap[2]
    out[classimg == 3.0] = colormap[3]
    out[classimg == 4.0] = colormap[4]

    return out


if __name__=="__main__":
    trainData = hs.getData(filename='D:\-_Hyper_Spec_-\HYPER_SPEC_TRAIN.h5', dat_idx=25*49, lab_idx=49)
    #testData = hs.getData(filename='D:\-_Hyper_Spec_-\HYPER_SPEC_TRAIN.h5')




    app = QtGui.QApplication(sys.argv)
    form = HyperSpecApp()
    form.show()
    form.update() #start with something
    #print("TRAIN " + str(np.shape(trainData['dcb'])))
    img = trainData['dcb'][:, :, 0:25]  #fromn red test 343:370
    img1 = np.swapaxes(img, 2, 0)

    form.ImageView2.setImage(img1)
    form.ImageView2.export("Pre_FFT_Masking_.png")
    #create FFT Plot for paper
    fft_example = hs.hsfft(img1)
    log_fft = np.log2(fft_example)
    aaa = ft.fftshift(log_fft.real)
    form.ImageView3.setImage(aaa) #levels=[np.amin(fft_example.real), np.amax(fft_example.real)+.01])
    form.ImageView3.export("FFT_DCB_.png")
    #v89 = imshow(aaa)
    #aaaa = ImageView.set_data(aaa)

    mask = hs.genMask(offset=41)
    #form.ImageView3.setImage(mask, levels=[np.amin(mask),np.amax(mask)+.0001])
    #ImageView doesn't seem to display binary arrays very well so add a small value.
    out_dcb = hs.dcbFilter(img)
    form.ImageView1.setImage(out_dcb.real)
    form.ImageView1.export("Post_FFT_Masking_.png")
    gtbatch = adjustLabels(trainData['classLabels'])
    gt = gtbatch
    #form.ImageView2.setImage(gt)

    t = np.swapaxes(out_dcb, 0, 2)
    t = np.swapaxes(t, 0, 1)
    fftImg = t.real.astype(np.float32, copy=False)
    print('SHAPE OF INPUT IMG: ' + str(np.shape(img)))
    print('SHAPE OF FFT OUT: ' + str(np.shape(fftImg)))


    (m, c) = runKmeans(form.pltView1, fftImg)
    (mm, cc) = runKmeans(form.pltView2, img)

    view_cube(fftImg)

    (raw_results, raw_Errors) = runSpectral(img, gt, title="Raw")
    (fft_results, fft_Errors) = runSpectral(fftImg, gt, title="FFT")

    (raw_pc_results, raw_pc_Errors, raw_pc) = runPCA(img, gt, title="Raw")
    (fft_pc_results, fft_pc_Errors, fft_pc) = runPCA(fftImg, gt, title="FFT")
    print('SHAPE of results: ' + str(np.shape(fft_pc_results)))
    print(fft_pc_results)
    xdata = fft_pc.transform(fftImg)
    w = view_nd(xdata[:, :, :5], classes=gt.astype(np.int8, copy=False), title="FFT_DCB PCA Components")

    ydata = fft_pc.transform(img)
    w = view_nd(ydata[:, :, :5], classes=gt.astype(np.int8, copy=False), title="DCB PCA Components")


    # perform mathematical morphology operations to reduce noise in results
    # convert each class to binary images then recombine a the end
    rbc_img = fft_pc_results == 2.0
    wbc_img = fft_pc_results == 1.0
    nuc_img = fft_pc_results == 3.0
    bkg_img = fft_pc_results == 4.0


    (wbc_o, wbc_c) = cleanResults(wbc_img)
    (rbc_o, rbc_c) = cleanResults(rbc_img)
    (nuc_o, nuc_c) = cleanResults(nuc_img)
    (bkg_o, bkg_c) = cleanResults(bkg_img)


    #open_rbc = ndimage.binary_opening(rbc_img)
    #clse_rbc = ndimage.binary_closing(open_rbc)
    #print(rbc_img)


    def calcAccuracy(bin, gt, c):
        '''takes binary image of one class and compares it to the ground
        truth of that class.  Error is calculated based on weighted empirical error'''
        class_gt = gt[gt == c]
        class_match = class_gt[bin]
        class_err = class_match


    ti = combineLabels(rbc_o, wbc_o, nuc_o, bkg_img)
    color = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.ubyte)
    tis = create_rgb(ti, color)

    #calculate accuracy for each class.


    v6876 = imshow(tis, title="Cleaned FFT PCA GT Result")
    pylab.savefig("Cleaned FFT PCA GT Result.png", bbox_inches='tight')
    #form.ImageView3.setImage(rbc_c)
    #form.ImageView1.setImage(wbc_c)
    #form.ImageView2.setImage(ti)

    fft_classes = create_training_classes(fftImg, gt, True)
    fft_means = np.zeros((len(fft_classes), fftImg.shape[2]), float)

    for (e, g) in enumerate(fft_classes):
        fft_means[e] = g.stats.mean

    fft_angles = spectral_angles(fftImg, fft_means)
    fft_clmap = np.argmin(fft_angles, 2)
    v20 = imshow(classes=((fft_clmap + 1) * (gt != 0)))




    #v9 = plt.imshow(rbc_img) #, title="RBC results")
    #pylab.savefig(("RBC_results.png"), bbox_inches='tight')
    #v10 = imshow(classes=open_rbc, title="RBC open_results")
    #pylab.savefig(("RBC_open_results.png"), bbox_inches='tight')
    #v12 = imshow(classes=clse_rbc, title="RBC_closed results")
    #pylab.savefig(("RBC_closed_results.png"), bbox_inches='tight')

    #subplot = form.matWidget0.getFigure().imshow(clmap)
    #form.matWidget0.

    #STEPS = np.array([0, 1, 2, 3, 4])
    #CLRS = ['k', 'r', 'y', 'b', 'w']
    #clrmp = pyqtgraph.ColorMap(STEPS, np.array([pyqtgraph.colorTuple(pyqtgraph.Color(c)) for c in CLRS]))
    #print(clrmp)
    #form.ImageView2.setImage(clmap)

    #form.ImageView2.ui.histogram.gradient.setColorMap(clrmp)

    #form.img.setImage(clmap,auto_levels=False)
    #form.img.show()

    app.exec_()
    print("DONE")