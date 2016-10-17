__author__ = 'Christo Robison'

'''This class is a Thread wrapper for loading data in to another app
    in a background process to keep GUI from hanging'''

from PyQt4 import QtCore, QtGui
import hs_imFFTW as hs
import numpy as np



class dataLoader(QtCore.QThread):
    #custom signal made to emit numpy array
    loadSignal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, dataPath):
        QtCore.QThread.__init__(self)
        self.dataPath = dataPath

    def __del__(self):
        self.wait()

    def _load_data(self, dataPath):
        data = hs.getData(dataPath)
        return data

    def run(self):
        '''tell the thread to laod data with given path and send loaded data to
            main thread.'''
        hs_data=self._load_data(self.dataPath)
        self.loadSignal.emit(hs_data)
        self.sleep(2)

