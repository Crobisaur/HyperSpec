__author__ = "Christo Robison"

from PyQt4 import QtGui, QtCore

class Worker(QtCore.QObject):
    threadInfo = QtCore.pyqtSignal(object, object)

    @QtCore.pyqtSlot()
    def emitInfo(self):
        self.threadInfo.emit(self.objectName(), QtCore.QThread.currentThreadId())


def handleShowThreads(name, id):
    print('Main: %s' % QtCore.QThread.currentThreadId())
    print('%s: %s\n' % (name, id))


if __name__ == '__main__':
    import sys
    thread = QtCore.QThread()
    worker1 = Worker()
    worker1.setObjectName('Worker1')
    worker1.moveToThread(thread)
    worker1.emitInfo
    worker1.threadInfo.connect(handleShowThreads(wname, wid))
    thread.start()
