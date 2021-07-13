# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'celltypedlg.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_CellTypeDlg(object):

    def setupUi(self, CellTypeDlg):

        CellTypeDlg.setObjectName("CellTypeDlg")

        CellTypeDlg.resize(381, 365)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(CellTypeDlg)

        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.cellTypeTable = QtWidgets.QTableWidget(CellTypeDlg)

        self.cellTypeTable.setEnabled(True)

        self.cellTypeTable.setBaseSize(QtCore.QSize(256, 171))

        self.cellTypeTable.setObjectName("cellTypeTable")

        self.cellTypeTable.setColumnCount(2)

        self.cellTypeTable.setRowCount(0)

        item = QtWidgets.QTableWidgetItem()

        self.cellTypeTable.setHorizontalHeaderItem(0, item)

        item = QtWidgets.QTableWidgetItem()

        self.cellTypeTable.setHorizontalHeaderItem(1, item)

        self.verticalLayout_2.addWidget(self.cellTypeTable)

        self.verticalLayout = QtWidgets.QVBoxLayout()

        self.verticalLayout.setObjectName("verticalLayout")

        self.horizontalLayout = QtWidgets.QHBoxLayout()

        self.horizontalLayout.setObjectName("horizontalLayout")

        self.label_4 = QtWidgets.QLabel(CellTypeDlg)

        self.label_4.setObjectName("label_4")

        self.horizontalLayout.addWidget(self.label_4)

        self.cellTypeLE = QtWidgets.QLineEdit(CellTypeDlg)

        self.cellTypeLE.setObjectName("cellTypeLE")

        self.horizontalLayout.addWidget(self.cellTypeLE)

        self.freezeCHB = QtWidgets.QCheckBox(CellTypeDlg)

        self.freezeCHB.setObjectName("freezeCHB")

        self.horizontalLayout.addWidget(self.freezeCHB)

        self.cellTypeAddPB = QtWidgets.QPushButton(CellTypeDlg)

        self.cellTypeAddPB.setObjectName("cellTypeAddPB")

        self.horizontalLayout.addWidget(self.cellTypeAddPB)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.clearCellTypeTablePB = QtWidgets.QPushButton(CellTypeDlg)

        self.clearCellTypeTablePB.setObjectName("clearCellTypeTablePB")

        self.horizontalLayout_2.addWidget(self.clearCellTypeTablePB)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(spacerItem)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_3.setObjectName("horizontalLayout_3")

        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(spacerItem1)

        self.okPB = QtWidgets.QPushButton(CellTypeDlg)

        self.okPB.setObjectName("okPB")

        self.horizontalLayout_3.addWidget(self.okPB)

        self.cancelPB = QtWidgets.QPushButton(CellTypeDlg)

        self.cancelPB.setObjectName("cancelPB")

        self.horizontalLayout_3.addWidget(self.cancelPB)

        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.cellTypeTable.raise_()

        self.clearCellTypeTablePB.raise_()

        self.label_4.raise_()

        self.cellTypeAddPB.raise_()

        self.cellTypeLE.raise_()

        self.freezeCHB.raise_()

        self.okPB.raise_()

        self.cancelPB.raise_()



        self.retranslateUi(CellTypeDlg)

        self.okPB.clicked.connect(CellTypeDlg.accept)

        self.cancelPB.clicked.connect(CellTypeDlg.reject)

        QtCore.QMetaObject.connectSlotsByName(CellTypeDlg)



    def retranslateUi(self, CellTypeDlg):

        _translate = QtCore.QCoreApplication.translate

        CellTypeDlg.setWindowTitle(_translate("CellTypeDlg", "Please define cell types"))

        item = self.cellTypeTable.horizontalHeaderItem(0)

        item.setText(_translate("CellTypeDlg", "Cell Type"))

        item = self.cellTypeTable.horizontalHeaderItem(1)

        item.setText(_translate("CellTypeDlg", "Freeze"))

        self.label_4.setText(_translate("CellTypeDlg", "Cell Type"))

        self.freezeCHB.setToolTip(_translate("CellTypeDlg", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"

"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"

"p, li { white-space: pre-wrap; }\n"

"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"

"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Indicates whether cells of this type should remain frozen during simulation</span></p></body></html>"))

        self.freezeCHB.setText(_translate("CellTypeDlg", "Freeze"))

        self.cellTypeAddPB.setText(_translate("CellTypeDlg", "Add"))

        self.clearCellTypeTablePB.setText(_translate("CellTypeDlg", "Clear Table"))

        self.okPB.setText(_translate("CellTypeDlg", "OK"))

        self.cancelPB.setText(_translate("CellTypeDlg", "Cancel"))



