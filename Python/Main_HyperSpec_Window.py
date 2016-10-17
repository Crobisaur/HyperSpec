__author__ = 'Christo Robison'

from PyQt4 import QtGui, QtCore
import sys
import HyperSpecGui
import numpy as np
import h5py
#from scipy import ndimage
#from scipy import fftpack as ft
#import matplotlib
#matplotlib.use('QT4Agg')
#import matplotlib.pylab as pylab
#import matplotlib.pyplot as plt

import time
#import pyqtgraph
#import hs_imFFTW as hs
import dataLoader


'''
background = 0
red = 1 WBC
green = 2 RBC
pink = 3 nuclear material
yellow = 4 ignore'''

class HyperSpecApp(QtGui.QMainWindow, HyperSpecGui.Ui_MainWindow):
    def __init__(self, parent=None):
        #pyqtgraph.setConfigOption('background', 'k')  #set background color before loading widget
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pltView1.plotItem.showGrid(True, True, 0.7)
        self.pltView2.plotItem.showGrid(True, True, 0.7)
        self.updateBtn.clicked.connect(self.loadDataCubes()) #use this button to load data

    #def _selectFile(self):
    #    dataPath = QtGui.QFileDialog.getOpenFileName()
    #    return dataPath

    def loadDataCubes(self, dataPath = QtGui.QFileDialog.getOpenFileName()):
        # using an input string from user, load that data
        self.get_thread = dataLoader(dataPath)
        self.connect(self.get_thread, self.handleSignal)
        self.connect(self.get_thread, QtCore.SIGNAL("finished()"), self.done)
        self.get_thread.start()
        self.updateBtn.setEnabled(False)

    def handleSignal(self, data):
        self.hs_data = data
        print(str(np.shape(self.hs_data)))

    def done(self):
        QtGui.QMessageBox.information(self, "Done!", "Done loading data!")

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






# def runKmeans(view, inputImg, clusters=6, iters=300):
#     (m, c) = kmeans(inputImg, clusters, iters)
#     #mw = pyqtgraph.widgets.MatplotlibWidget.MatplotlibWidget()
#     #subplot = form.pltView0.add_subplot(111)
#     #form.pltView0.
#     for i in range(c.shape[0]):
#         view.plot(c[i])
#     #form.pltView0.plot(subplot)
#     return (m, c)
#
# def adjustLabels(dcb, bkgnd=0.0, unknown=4.0):
#     dcb[dcb==0.0] = 5.0
#     dcb[dcb==4.0] = 0.0
#     dcb[dcb==5.0] = 4.0
#     return dcb
#
# def runSpectral(dcb, gt, title = 'dcb'):
#     (classes, gmlc, clmap) = runGauss(dcb, gt)
#     (gtresults, gtErrors) = genResults(clmap, gt)
#     displayPlots(clmap, gt, gtresults, gtErrors, (title+" Gaussian Classifer"))
#     return (gtresults, gtErrors)
#
# def runPCA(dcb, gt, title = 'dcb'):
#     pc = principal_components(dcb)
#     pc_0999 = pc.reduce(fraction=0.999)
#     img_pc = pc_0999.transform(dcb)
#     (classes, gmlc, clmap) = runGauss(img_pc, gt)
#     (gtresults, gtErrors) = genResults(clmap, gt)
#     displayPlots(clmap, gt, gtresults, gtErrors, (title+" PCA Gaussian Classifer"))
#     return (gtresults, gtErrors, pc)
#
# def genResults(clmap, gt):
#     gtresults = clmap * (gt!=0)
#     gtErrors = gtresults * (gtresults !=gt)
#     return (gtresults, gtErrors)
#
# def runGauss(dcb, gt):
#     classes = create_training_classes(dcb, gt)
#     gmlc = GaussianClassifier(classes, min_samples=200)
#     clmap = gmlc.classify_image(dcb)
#     return (classes, gmlc, clmap)
#
# def displayPlots(clmap, gt, gtresults = None, gtErrors = None, title = 'classifier'):
#     if (gtresults is None and gtErrors is None):
#         (gtresults, gtErrors) = genResults(clmap, gt)
#     v0 = imshow(classes=clmap, title=(title+" results"))
#     pylab.savefig((title+" results.png"), bbox_inches='tight')
#     v1 = imshow(classes = gtresults, title=(title+" gt Results"))
#     pylab.savefig((title + " gt Results.png"), bbox_inches='tight')
#     v2 = imshow(classes = gtErrors, title=(title+" Error"))
#     pylab.savefig((title + " Error.png"), bbox_inches='tight')
#
# def cleanResults(inputImg, cls_iter=1, open_iter=1):
#     open = ndimage.binary_opening(inputImg, iterations=open_iter)
#     close = ndimage.binary_opening(inputImg, iterations=cls_iter)
#     return (open, close)
#
#
# def combineLabels(rbc, wbc, nuc, bkgd):
#     out = np.zeros(np.shape(rbc), dtype=np.float64)
#     out[bkgd == 1] = 4.0
#     out[rbc == 1] = 2.0
#     out[wbc == 1] = 1.0
#     out[nuc == 1] = 3.0
#     out[out == 0] = 0.0
#
#     print(out)
#     return out
#
# def create_rgb(classimg, colormap=None):
#     if colormap is None:
#         colormap = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [255, 0, 255], [255, 255, 0]], dtype=np.ubyte)
#     h,w = np.shape(classimg)
#     out = np.zeros([h, w, 3], dtype=np.uint8)
#     out[classimg == 0.0] = colormap[0]
#     out[classimg == 1.0] = colormap[1]
#     out[classimg == 2.0] = colormap[2]
#     out[classimg == 3.0] = colormap[3]
#     out[classimg == 4.0] = colormap[4]
#
#     return out


def main():
    app = QtGui.QApplication(sys.argv)
    form = HyperSpecApp()
    form.show()
    app._exec()

if __name__=="__main__":
    #trainData = hs.getData(filename='D:\-_Hyper_Spec_-\HYPER_SPEC_TRAIN.h5', dat_idx=25*49, lab_idx=49)
    #testData = hs.getData(filename='D:\-_Hyper_Spec_-\HYPER_SPEC_TRAIN.h5')
    main()

    #app = QtGui.QApplication(sys.argv)
    #form = HyperSpecApp()
    #form.show()
    #form.update() #start with something
    #print("TRAIN " + str(np.shape(trainData['dcb'])))
    #img = trainData['dcb'][:, :, 0:25]  #fromn red test 343:370
    #img1 = np.swapaxes(img, 2, 0)
