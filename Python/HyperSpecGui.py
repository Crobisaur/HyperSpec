# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\-_Google_Drive_-\-_Thesis_-\GUIs\HyperSpec\Version_1_1\HyperSpec_1_1.ui'
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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1363, 913)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.ImageView1 = ImageView(self.centralwidget)
        self.ImageView1.setGeometry(QtCore.QRect(40, 30, 611, 451))
        self.ImageView1.setObjectName(_fromUtf8("ImageView1"))
        self.ImageView2 = ImageView(self.centralwidget)
        self.ImageView2.setGeometry(QtCore.QRect(670, 30, 651, 451))
        self.ImageView2.setObjectName(_fromUtf8("ImageView2"))
        self.ImageView3 = ImageView(self.centralwidget)
        self.ImageView3.setGeometry(QtCore.QRect(40, 500, 611, 361))
        self.ImageView3.setObjectName(_fromUtf8("ImageView3"))
        self.ImageView4 = ImageView(self.centralwidget)
        self.ImageView4.setGeometry(QtCore.QRect(670, 500, 651, 361))
        self.ImageView4.setObjectName(_fromUtf8("ImageView4"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1363, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtGui.QAction(MainWindow)
        self.actionExit.setObjectName(_fromUtf8("actionExit"))
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.actionExit.setText(_translate("MainWindow", "Exit", None))

from pyqtgraph import ImageView

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

