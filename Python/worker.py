__author__ = "Christo Robison"

from PyQt4 import QtGui, QtCore

class Worker(QtCore.QObject):
    threadInfo = QtCore.pyqtSignal(object, object)

    @QtCore.pyqtSlot()
    def emitInfo(self):
        self.threadInfo.emit(self.objectName(), QtCore.QThread.currentThreadId())

if __name__ == '__main__':
    import sys