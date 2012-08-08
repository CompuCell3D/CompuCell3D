# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gotolinedlg.ui'
#
# Created: Mon May 31 11:01:10 2010
#      by: PyQt4 UI code generator 4.7.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_GoToLineDlg(object):
    def setupUi(self, GoToLineDlg):
        GoToLineDlg.setObjectName("GoToLineDlg")
        GoToLineDlg.resize(338, 115)
        self.horizontalLayout_2 = QtGui.QHBoxLayout(GoToLineDlg)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.goToLineLabel = QtGui.QLabel(GoToLineDlg)
        self.goToLineLabel.setObjectName("goToLineLabel")
        self.horizontalLayout.addWidget(self.goToLineLabel)
        self.goToLineEdit = QtGui.QLineEdit(GoToLineDlg)
        self.goToLineEdit.setObjectName("goToLineEdit")
        self.horizontalLayout.addWidget(self.goToLineEdit)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.line = QtGui.QFrame(GoToLineDlg)
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_2.addWidget(self.line)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.goButton = QtGui.QPushButton(GoToLineDlg)
        self.goButton.setObjectName("goButton")
        self.verticalLayout_2.addWidget(self.goButton)
        spacerItem1 = QtGui.QSpacerItem(17, 13, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.closeButton = QtGui.QPushButton(GoToLineDlg)
        self.closeButton.setObjectName("closeButton")
        self.verticalLayout_2.addWidget(self.closeButton)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.goToLineLabel.setBuddy(self.goToLineEdit)

        self.retranslateUi(GoToLineDlg)
        QtCore.QObject.connect(self.closeButton, QtCore.SIGNAL("clicked()"), GoToLineDlg.reject)
        QtCore.QMetaObject.connectSlotsByName(GoToLineDlg)

    def retranslateUi(self, GoToLineDlg):
        GoToLineDlg.setWindowTitle(QtGui.QApplication.translate("GoToLineDlg", "Go To Line", None, QtGui.QApplication.UnicodeUTF8))
        self.goToLineLabel.setText(QtGui.QApplication.translate("GoToLineDlg", "Go to line...", None, QtGui.QApplication.UnicodeUTF8))
        self.goButton.setText(QtGui.QApplication.translate("GoToLineDlg", "Go", None, QtGui.QApplication.UnicodeUTF8))
        self.closeButton.setText(QtGui.QApplication.translate("GoToLineDlg", "Close", None, QtGui.QApplication.UnicodeUTF8))

