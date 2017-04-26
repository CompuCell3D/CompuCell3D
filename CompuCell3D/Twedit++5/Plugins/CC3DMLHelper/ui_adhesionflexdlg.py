# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'adhesionflexdlg.ui'
#
# Created: Thu May 10 14:05:25 2012
#      by: PyQt4 UI code generator 4.8.6
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_AdhesionFlexDlg(object):
    def setupUi(self, AdhesionFlexDlg):
        AdhesionFlexDlg.setObjectName(_fromUtf8("AdhesionFlexDlg"))
        AdhesionFlexDlg.resize(307, 298)
        AdhesionFlexDlg.setWindowTitle(QtGui.QApplication.translate("AdhesionFlexDlg", "Please define adhesion molecules", None, QtGui.QApplication.UnicodeUTF8))
        self.verticalLayout_2 = QtGui.QVBoxLayout(AdhesionFlexDlg)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.afTable = QtGui.QTableWidget(AdhesionFlexDlg)
        self.afTable.setEnabled(True)
        self.afTable.setBaseSize(QtCore.QSize(256, 171))
        self.afTable.setObjectName(_fromUtf8("afTable"))
        self.afTable.setColumnCount(1)
        self.afTable.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        item.setText(QtGui.QApplication.translate("AdhesionFlexDlg", "Adhesion Molecule", None, QtGui.QApplication.UnicodeUTF8))
        self.afTable.setHorizontalHeaderItem(0, item)
        self.verticalLayout_2.addWidget(self.afTable)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_9 = QtGui.QLabel(AdhesionFlexDlg)
        self.label_9.setText(QtGui.QApplication.translate("AdhesionFlexDlg", "Molecule", None, QtGui.QApplication.UnicodeUTF8))
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.horizontalLayout.addWidget(self.label_9)
        self.afMoleculeLE = QtGui.QLineEdit(AdhesionFlexDlg)
        self.afMoleculeLE.setToolTip(QtGui.QApplication.translate("AdhesionFlexDlg", "Specify names of the adhesion molecules you want to use int he simulation", None, QtGui.QApplication.UnicodeUTF8))
        self.afMoleculeLE.setText(_fromUtf8(""))
        self.afMoleculeLE.setObjectName(_fromUtf8("afMoleculeLE"))
        self.horizontalLayout.addWidget(self.afMoleculeLE)
        self.afMoleculeAddPB = QtGui.QPushButton(AdhesionFlexDlg)
        self.afMoleculeAddPB.setText(QtGui.QApplication.translate("AdhesionFlexDlg", "Add", None, QtGui.QApplication.UnicodeUTF8))
        self.afMoleculeAddPB.setObjectName(_fromUtf8("afMoleculeAddPB"))
        self.horizontalLayout.addWidget(self.afMoleculeAddPB)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_10 = QtGui.QLabel(AdhesionFlexDlg)
        self.label_10.setText(QtGui.QApplication.translate("AdhesionFlexDlg", "Binding Formula", None, QtGui.QApplication.UnicodeUTF8))
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.horizontalLayout_2.addWidget(self.label_10)
        self.bindingFormulaLE = QtGui.QLineEdit(AdhesionFlexDlg)
        self.bindingFormulaLE.setToolTip(QtGui.QApplication.translate("AdhesionFlexDlg", "This is binary function that takes atwo arguments -  Molecule1 and Molecule2. The allowed functions are those given by muParser - see http://muparser.sourceforge.net/", None, QtGui.QApplication.UnicodeUTF8))
        self.bindingFormulaLE.setText(QtGui.QApplication.translate("AdhesionFlexDlg", "min(Molecule1,Molecule2)", None, QtGui.QApplication.UnicodeUTF8))
        self.bindingFormulaLE.setObjectName(_fromUtf8("bindingFormulaLE"))
        self.horizontalLayout_2.addWidget(self.bindingFormulaLE)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.clearAFTablePB = QtGui.QPushButton(AdhesionFlexDlg)
        self.clearAFTablePB.setText(QtGui.QApplication.translate("AdhesionFlexDlg", "Clear Table", None, QtGui.QApplication.UnicodeUTF8))
        self.clearAFTablePB.setObjectName(_fromUtf8("clearAFTablePB"))
        self.horizontalLayout_4.addWidget(self.clearAFTablePB)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.okPB = QtGui.QPushButton(AdhesionFlexDlg)
        self.okPB.setText(QtGui.QApplication.translate("AdhesionFlexDlg", "OK", None, QtGui.QApplication.UnicodeUTF8))
        self.okPB.setObjectName(_fromUtf8("okPB"))
        self.horizontalLayout_3.addWidget(self.okPB)
        self.cancelPB = QtGui.QPushButton(AdhesionFlexDlg)
        self.cancelPB.setText(QtGui.QApplication.translate("AdhesionFlexDlg", "Cancel", None, QtGui.QApplication.UnicodeUTF8))
        self.cancelPB.setObjectName(_fromUtf8("cancelPB"))
        self.horizontalLayout_3.addWidget(self.cancelPB)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.retranslateUi(AdhesionFlexDlg)
        QtCore.QObject.connect(self.okPB, QtCore.SIGNAL(_fromUtf8("clicked()")), AdhesionFlexDlg.accept)
        QtCore.QObject.connect(self.cancelPB, QtCore.SIGNAL(_fromUtf8("clicked()")), AdhesionFlexDlg.reject)
        QtCore.QMetaObject.connectSlotsByName(AdhesionFlexDlg)

    def retranslateUi(self, AdhesionFlexDlg):
        item = self.afTable.horizontalHeaderItem(0)

