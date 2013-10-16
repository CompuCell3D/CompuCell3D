# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\parameterdlg.ui'
#
# Created: Wed Oct 16 18:17:23 2013
#      by: PyQt4 UI code generator 4.8.6
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_ParameterDlg(object):
    def setupUi(self, ParameterDlg):
        ParameterDlg.setObjectName(_fromUtf8("ParameterDlg"))
        ParameterDlg.resize(423, 293)
        ParameterDlg.setWindowTitle(QtGui.QApplication.translate("ParameterDlg", "Scannable Paramerters", None, QtGui.QApplication.UnicodeUTF8))
        self.verticalLayout = QtGui.QVBoxLayout(ParameterDlg)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.elementLB = QtGui.QLabel(ParameterDlg)
        self.elementLB.setText(QtGui.QApplication.translate("ParameterDlg", "Element:", None, QtGui.QApplication.UnicodeUTF8))
        self.elementLB.setObjectName(_fromUtf8("elementLB"))
        self.horizontalLayout_2.addWidget(self.elementLB)
        self.elemLE = QtGui.QLineEdit(ParameterDlg)
        self.elemLE.setReadOnly(True)
        self.elemLE.setObjectName(_fromUtf8("elemLE"))
        self.horizontalLayout_2.addWidget(self.elemLE)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.paramTW = QtGui.QTableWidget(ParameterDlg)
        self.paramTW.setObjectName(_fromUtf8("paramTW"))
        self.paramTW.setColumnCount(4)
        self.paramTW.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        item.setText(QtGui.QApplication.translate("ParameterDlg", "Parameter", None, QtGui.QApplication.UnicodeUTF8))
        self.paramTW.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        item.setText(QtGui.QApplication.translate("ParameterDlg", "Type", None, QtGui.QApplication.UnicodeUTF8))
        self.paramTW.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        item.setText(QtGui.QApplication.translate("ParameterDlg", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.paramTW.setHorizontalHeaderItem(2, item)
        item = QtGui.QTableWidgetItem()
        item.setText(QtGui.QApplication.translate("ParameterDlg", "Action", None, QtGui.QApplication.UnicodeUTF8))
        self.paramTW.setHorizontalHeaderItem(3, item)
        self.verticalLayout.addWidget(self.paramTW)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.pushButton = QtGui.QPushButton(ParameterDlg)
        self.pushButton.setText(QtGui.QApplication.translate("ParameterDlg", "Cancel", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.horizontalLayout.addWidget(self.pushButton)
        self.okPB = QtGui.QPushButton(ParameterDlg)
        self.okPB.setText(QtGui.QApplication.translate("ParameterDlg", "OK", None, QtGui.QApplication.UnicodeUTF8))
        self.okPB.setObjectName(_fromUtf8("okPB"))
        self.horizontalLayout.addWidget(self.okPB)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(ParameterDlg)
        QtCore.QObject.connect(self.okPB, QtCore.SIGNAL(_fromUtf8("clicked()")), ParameterDlg.accept)
        QtCore.QObject.connect(self.pushButton, QtCore.SIGNAL(_fromUtf8("clicked()")), ParameterDlg.reject)
        QtCore.QMetaObject.connectSlotsByName(ParameterDlg)

    def retranslateUi(self, ParameterDlg):
        item = self.paramTW.horizontalHeaderItem(0)
        item = self.paramTW.horizontalHeaderItem(1)
        item = self.paramTW.horizontalHeaderItem(2)
        item = self.paramTW.horizontalHeaderItem(3)

