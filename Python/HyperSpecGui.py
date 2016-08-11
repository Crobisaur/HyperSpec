# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\-_Google_Drive_-\-_Thesis_-\GUIs\HyperSpec\Version_1_1\HyperSpec_1_4.ui'
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
        MainWindow.resize(1341, 925)
        MainWindow.setFocusPolicy(QtCore.Qt.NoFocus)
        MainWindow.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        MainWindow.setStyleSheet(_fromUtf8("background-color: rgb(60,60,60); QMenuBar{ background-color: rgb(60,60,60)} QMenuBar::Item{background: transparent}; QMenu::Item{background-color: rgb(60,60,60)}; QPushButton{background-color: rgb(A4,A4,A4)};\n"
""))
        MainWindow.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.USVirginIslands))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setStyleSheet(_fromUtf8("background-color: rgb(60,60,60)"))
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.pltView1 = PlotWidget(self.centralwidget)
        self.pltView1.setGeometry(QtCore.QRect(40, 10, 401, 451))
        self.pltView1.setObjectName(_fromUtf8("pltView1"))
        self.ImageView1 = ImageView(self.centralwidget)
        self.ImageView1.setGeometry(QtCore.QRect(40, 480, 611, 361))
        self.ImageView1.setObjectName(_fromUtf8("ImageView1"))
        self.ImageView2 = ImageView(self.centralwidget)
        self.ImageView2.setGeometry(QtCore.QRect(670, 480, 651, 361))
        self.ImageView2.setObjectName(_fromUtf8("ImageView2"))
        self.updateBtn = QtGui.QPushButton(self.centralwidget)
        self.updateBtn.setGeometry(QtCore.QRect(40, 850, 75, 23))
        self.updateBtn.setStyleSheet(_fromUtf8("background-color: rgb(117, 117, 117);"))
        self.updateBtn.setObjectName(_fromUtf8("updateBtn"))
        self.pltView2 = PlotWidget(self.centralwidget)
        self.pltView2.setGeometry(QtCore.QRect(460, 10, 441, 451))
        self.pltView2.setObjectName(_fromUtf8("pltView2"))
        self.ImageView3 = ImageView(self.centralwidget)
        self.ImageView3.setGeometry(QtCore.QRect(920, 10, 401, 451))
        self.ImageView3.setObjectName(_fromUtf8("ImageView3"))
        self.openBtn = QtGui.QPushButton(self.centralwidget)
        self.openBtn.setGeometry(QtCore.QRect(130, 850, 75, 23))
        self.openBtn.setStyleSheet(_fromUtf8("background-color: rgb(117, 117, 117);"))
        self.openBtn.setObjectName(_fromUtf8("openBtn"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1341, 21))
        font = QtGui.QFont()
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.menubar.setFont(font)
        self.menubar.setAutoFillBackground(False)
        self.menubar.setStyleSheet(_fromUtf8(""))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setStyleSheet(_fromUtf8("background-color: rgb(97, 97, 97);"))
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
        MainWindow.setWindowTitle(_translate("MainWindow", "HyperSpec", None))
        self.updateBtn.setText(_translate("MainWindow", "Update", None))
        self.openBtn.setText(_translate("MainWindow", "Open", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.actionExit.setText(_translate("MainWindow", "Exit", None))

from pyqtgraph import ImageView, PlotWidget

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

