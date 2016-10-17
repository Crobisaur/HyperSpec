from PyQt4 import QtGui, QtCore
import Python.worker

class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.button = QtGui.QPushButton('Test', self)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.button)
        self.te = QtGui.QTextEdit()
        layout.addWidget(self.te)
        self.thread = QtCore.QThread(self)
        self.worker1 = Python.worker.Worker()
        self.worker1.setObjectName('Worker1')
        self.worker1.moveToThread(self.thread)
        self.worker1.threadInfo.connect(self.handleShowThreads)
        self.button.clicked.connect(self.worker1.emitInfo)
        self.thread.start()

    def handleShowThreads(self, name, identifier):
        print('Main: %s' % QtCore.QThread.currentThreadId())
        self.te.append('Main: %s' % QtCore.QThread.currentThreadId())
        print('%s: %s\n' % (name, identifier))
        self.te.append('%s: %s\n' % (name, identifier))
        self.te.append(QtGui.QFileDialog.getOpenFileName())

    def closeEvent(self, event):
        self.thread.quit()
        self.thread.wait()


if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())