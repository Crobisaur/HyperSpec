# -*- coding: utf-8 -*-
#Default encoding from pyuic4 is broken, need to manually change to utf-8

# Form implementation generated from reading ui file 'D:\-_Google_Drive_-\-_Thesis_-\GUIs\HyperSpec\Version_1_0\HyperSpec_1_0.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_HyperSpec_MainWindow(object):
    def setupUi(self, HyperSpec_MainWindow):
        HyperSpec_MainWindow.setObjectName(_fromUtf8("HyperSpec_MainWindow"))
        HyperSpec_MainWindow.resize(800, 752)
        self.centralwidget = QtGui.QWidget(HyperSpec_MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.plot1 = QtGui.QWidget(self.centralwidget)
        self.plot1.setGeometry(QtCore.QRect(10, 10, 381, 341))
        self.plot1.setObjectName(_fromUtf8("plot1"))
        self.plot2 = QtGui.QWidget(self.centralwidget)
        self.plot2.setGeometry(QtCore.QRect(400, 10, 391, 341))
        self.plot2.setObjectName(_fromUtf8("plot2"))
        self.plot3 = QtGui.QWidget(self.centralwidget)
        self.plot3.setGeometry(QtCore.QRect(10, 360, 381, 341))
        self.plot3.setObjectName(_fromUtf8("plot3"))
        self.plot4 = QtGui.QWidget(self.centralwidget)
        self.plot4.setGeometry(QtCore.QRect(400, 360, 391, 341))
        self.plot4.setObjectName(_fromUtf8("plot4"))
        HyperSpec_MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(HyperSpec_MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        HyperSpec_MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(HyperSpec_MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        HyperSpec_MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtGui.QAction(HyperSpec_MainWindow)
        self.actionExit.setObjectName(_fromUtf8("actionExit"))
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(HyperSpec_MainWindow)
        QtCore.QMetaObject.connectSlotsByName(HyperSpec_MainWindow)

    def retranslateUi(self, HyperSpec_MainWindow):
        HyperSpec_MainWindow.setWindowTitle(_translate("HyperSpec_MainWindow", "HyperSpec", None))
        self.menuFile.setTitle(_translate("HyperSpec_MainWindow", "File", None))
        self.actionExit.setText(_translate("HyperSpec_MainWindow", "Exit", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    HyperSpec_MainWindow = QtGui.QMainWindow()
    ui = Ui_HyperSpec_MainWindow()
    ui.setupUi(HyperSpec_MainWindow)
    HyperSpec_MainWindow.show()
    sys.exit(app.exec_())