__author__ = 'Christo Robison'

from PyQt4 import QtGui, QtCore
import sys
import HyperSpecGui
import numpy as np
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pylab as pylab

import time
import pyqtgraph
import hs_imFFTW as hs
from spectral import *
spectral.settings.WX_GL_DEPTH_SIZE = 16


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

        self.ImageView2.ui.histogram.gradient.setColorMap(clrmp)






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


if __name__=="__main__":
    trainData = hs.getData(filename='D:\-_Hyper_Spec_-\HYPER_SPEC_TEST_RED.h5')
    testData = hs.getData(filename='D:\-_Hyper_Spec_-\HYPER_SPEC_TEST_RED.h5')




    app = QtGui.QApplication(sys.argv)
    form = HyperSpecApp()
    form.show()
    form.update() #start with something
    img = trainData['dcb'][:, :, 343:370]
    img1 = np.swapaxes(img, 2, 0)

    form.ImageView1.setImage(img1)
    mask = hs.genMask(offset=11)
    #form.ImageView2.setImage(mask, levels=[np.amin(mask),np.amax(mask)+.0001])
    #ImageView doesn't seem to display binary arrays very well so add a small value.
    out_dcb = hs.dcbFilter(img)
    form.ImageView1.setImage(out_dcb.real)
    gtbatch = adjustLabels(trainData['classLabels'])
    gt = gtbatch[:, :, 11]
    #form.ImageView2.setImage(gt)

    t = np.swapaxes(out_dcb, 0, 2)
    t = np.swapaxes(t, 0, 1)
    fftImg = t.real.astype(np.float32, copy=False)
    print('SHAPE OF INPUT IMG: ' + str(np.shape(img)))
    print('SHAPE OF FFT OUT: ' + str(np.shape(fftImg)))


    #(m, c) = runKmeans(form.pltView1, testImg)

    view_cube(fftImg)

    (raw_results, raw_Errors) = runSpectral(img, gt, title="Raw")
    (fft_results, fft_Errors) = runSpectral(fftImg, gt, title="FFT")

    (raw_pc_results, raw_pc_Errors, raw_pc) = runPCA(img, gt, title="Raw")
    (fft_pc_results, fft_pc_Errors, fft_pc) = runPCA(fftImg, gt, title="FFT")
    xdata = fft_pc.transform(fftImg)
    w = view_nd(xdata[:, :, :10], classes=gt.astype(np.int8, copy=False), title="FFT_DCB PCA Components")

    ydata = fft_pc.transform(img)
    w = view_nd(ydata[:, :, :10], classes=gt.astype(np.int8, copy=False), title="DCB PCA Components")

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