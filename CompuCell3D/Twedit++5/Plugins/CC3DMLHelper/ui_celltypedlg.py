# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'celltypedlg.ui'
#
# Created: Thu May 10 14:05:18 2012
#      by: PyQt4 UI code generator 4.8.6
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_CellTypeDlg(object):
    def setupUi(self, CellTypeDlg):
        CellTypeDlg.setObjectName(_fromUtf8("CellTypeDlg"))
        CellTypeDlg.resize(381, 365)
        CellTypeDlg.setWindowTitle(QtGui.QApplication.translate("CellTypeDlg", "Please define cell types", None, QtGui.QApplication.UnicodeUTF8))
        self.verticalLayout_2 = QtGui.QVBoxLayout(CellTypeDlg)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.cellTypeTable = QtGui.QTableWidget(CellTypeDlg)
        self.cellTypeTable.setEnabled(True)
        self.cellTypeTable.setBaseSize(QtCore.QSize(256, 171))
        self.cellTypeTable.setObjectName(_fromUtf8("cellTypeTable"))
        self.cellTypeTable.setColumnCount(2)
        self.cellTypeTable.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        item.setText(QtGui.QApplication.translate("CellTypeDlg", "Cell Type", None, QtGui.QApplication.UnicodeUTF8))
        self.cellTypeTable.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        item.setText(QtGui.QApplication.translate("CellTypeDlg", "Freeze", None, QtGui.QApplication.UnicodeUTF8))
        self.cellTypeTable.setHorizontalHeaderItem(1, item)
        self.verticalLayout_2.addWidget(self.cellTypeTable)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_4 = QtGui.QLabel(CellTypeDlg)
        self.label_4.setText(QtGui.QApplication.translate("CellTypeDlg", "Cell Type", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout.addWidget(self.label_4)
        self.cellTypeLE = QtGui.QLineEdit(CellTypeDlg)
        self.cellTypeLE.setObjectName(_fromUtf8("cellTypeLE"))
        self.horizontalLayout.addWidget(self.cellTypeLE)
        self.freezeCHB = QtGui.QCheckBox(CellTypeDlg)
        self.freezeCHB.setToolTip(QtGui.QApplication.translate("CellTypeDlg", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Indicates whether cells of this type should remain frozen during simulation</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.freezeCHB.setText(QtGui.QApplication.translate("CellTypeDlg", "Freeze", None, QtGui.QApplication.UnicodeUTF8))
        self.freezeCHB.setObjectName(_fromUtf8("freezeCHB"))
        self.horizontalLayout.addWidget(self.freezeCHB)
        self.cellTypeAddPB = QtGui.QPushButton(CellTypeDlg)
        self.cellTypeAddPB.setText(QtGui.QApplication.translate("CellTypeDlg", "Add", None, QtGui.QApplication.UnicodeUTF8))
        self.cellTypeAddPB.setObjectName(_fromUtf8("cellTypeAddPB"))
        self.horizontalLayout.addWidget(self.cellTypeAddPB)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.clearCellTypeTablePB = QtGui.QPushButton(CellTypeDlg)
        self.clearCellTypeTablePB.setText(QtGui.QApplication.translate("CellTypeDlg", "Clear Table", None, QtGui.QApplication.UnicodeUTF8))
        self.clearCellTypeTablePB.setObjectName(_fromUtf8("clearCellTypeTablePB"))
        self.horizontalLayout_2.addWidget(self.clearCellTypeTablePB)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.okPB = QtGui.QPushButton(CellTypeDlg)
        self.okPB.setText(QtGui.QApplication.translate("CellTypeDlg", "OK", None, QtGui.QApplication.UnicodeUTF8))
        self.okPB.setObjectName(_fromUtf8("okPB"))
        self.horizontalLayout_3.addWidget(self.okPB)
        self.cancelPB = QtGui.QPushButton(CellTypeDlg)
        self.cancelPB.setText(QtGui.QApplication.translate("CellTypeDlg", "Cancel", None, QtGui.QApplication.UnicodeUTF8))
        self.cancelPB.setObjectName(_fromUtf8("cancelPB"))
        self.horizontalLayout_3.addWidget(self.cancelPB)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.retranslateUi(CellTypeDlg)
        QtCore.QObject.connect(self.okPB, QtCore.SIGNAL(_fromUtf8("clicked()")), CellTypeDlg.accept)
        QtCore.QObject.connect(self.cancelPB, QtCore.SIGNAL(_fromUtf8("clicked()")), CellTypeDlg.reject)
        QtCore.QMetaObject.connectSlotsByName(CellTypeDlg)

    def retranslateUi(self, CellTypeDlg):
        item = self.cellTypeTable.horizontalHeaderItem(0)
        item = self.cellTypeTable.horizontalHeaderItem(1)

