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

CHUNKSZ = 1024
FS = 44100 #Hz

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

def runSpectral(dcb, gt):
    (classes, gmlc, clmap) = runGauss(dcb, gt)
    (gtresults, gtErrors) = genResults(clmap, gt)
    displayPlots(clmap, gt, gtresults, gtErrors)
    return (gtresults, gtErrors)

def runPCA(dcb, gt):
    pc = principal_components(dcb)
    pc_0999 = pc.reduce(fraction=0.999)
    img_pc = pc_0999.transform(dcb)
    (classes, gmlc, clmap) = runGauss(img_pc, gt)
    (gtresults, gtErrors) = genResults(clmap, gt)
    displayPlots(clmap, gt, gtresults, gtErrors)
    return (gtresults, gtErrors)

def genResults(clmap, gt):
    gtresults = clmap * (gt!=0)
    gtErrors = gtresults * (gtresults !=gt)
    return (gtresults, gtErrors)

def runGauss(dcb, gt):
    classes = create_training_classes(dcb, gt)
    gmlc = GaussianClassifier(classes, min_samples=200)
    clmap = gmlc.classify_image(dcb)
    return (classes, gmlc, clmap)

def displayPlots(clmap, gt, gtresults = None, gtErrors = None):
    if (gtresults and gtErrors is None):
        (gtresults, gtErrors) = genResults(clmap, gt)
    v0 = imshow(classes=clmap)
    v1 = imshow(classes = gtresults)
    v2 = imshow(classes = gtErrors)


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
    testImg = t.real.astype(np.float32, copy=False)
    print('SHAPE OF INPUT IMG: ' + str(np.shape(img)))
    print('SHAPE OF FFT OUT: ' + str(np.shape(testImg)))


    #(m, c) = runKmeans(form.pltView1, testImg)

    # Display output from raw data
    classes_old = create_training_classes(img, gt)
    gmlc_old = GaussianClassifier(classes_old, min_samples=200)
    clmap_old = gmlc_old.classify_image(img)
    hh = imshow(testImg[:,:,3])
    g = imshow(classes=clmap_old)

    # Display output from fft data
    classes = create_training_classes(testImg, gt)
    gmlc = GaussianClassifier(classes, min_samples=200)
    clmap = gmlc.classify_image(testImg)
    print(clmap)
    v = imshow(classes=clmap)

    gtresults = clmap * (gt!=0)
    v = imshow(classes=gtresults)
    gtErrors = gtresults * (gtresults != gt)
    v = imshow(classes=gtErrors)

    # Perform PCA on filtered and unfiltered data

    pc_img = principal_components(img)
    pc_fimg = principal_components(testImg)
    v = imshow(pc_img.cov)
    v = imshow(pc_fimg.cov)
    pc_0999 = pc_img.reduce(fraction=0.999)
    pcf_0999 = pc_fimg.reduce(fraction=0.999)

    # Transform PCA data to useable format

    img_pc = pc_0999.transform(img)
    imgf_pc = pcf_0999.transform(testImg)

    # Show top 3 componenets for each data

    v = imshow(img_pc[:, :, :3], stretch_all=True)
    v = imshow(imgf_pc[:, :, :3], stretch_all=True)

    # Train Gaussian Classifier on new data

    classes_pc = create_training_classes(img_pc, gt)
    classes_pcf = create_training_classes(imgf_pc, gt)
    gmlc_pc = GaussianClassifier(classes_pc)
    gmlc_pcf = GaussianClassifier(classes_pcf)

    # Perform Gaussian Classifer on data

    clmap_pc = gmlc_pc.classify_image(img_pc)
    clmap_pcf = gmlc_pcf.classify_image(imgf_pc)

    # Show only trained data, remove anomalies

    clmap_pc_training = clmap_pc * (gt != 0)
    clmap_pcf_training = clmap_pcf * (gt != 0)

    # Show Classified images

    v = imshow(classes=clmap_pc_training)
    v = imshow(classes=clmap_pcf_training)

    # Show errors of classification

    pc_training_errors = clmap_pc_training * (clmap_pc_training != gt)
    pcf_training_errors = clmap_pcf_training * (clmap_pcf_training != gt)




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