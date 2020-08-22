# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'adhesionflexdlg.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_AdhesionFlexDlg(object):

    def setupUi(self, AdhesionFlexDlg):

        AdhesionFlexDlg.setObjectName("AdhesionFlexDlg")

        AdhesionFlexDlg.resize(307, 298)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(AdhesionFlexDlg)

        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.afTable = QtWidgets.QTableWidget(AdhesionFlexDlg)

        self.afTable.setEnabled(True)

        self.afTable.setBaseSize(QtCore.QSize(256, 171))

        self.afTable.setObjectName("afTable")

        self.afTable.setColumnCount(1)

        self.afTable.setRowCount(0)

        item = QtWidgets.QTableWidgetItem()

        self.afTable.setHorizontalHeaderItem(0, item)

        self.verticalLayout_2.addWidget(self.afTable)

        self.verticalLayout = QtWidgets.QVBoxLayout()

        self.verticalLayout.setObjectName("verticalLayout")

        self.horizontalLayout = QtWidgets.QHBoxLayout()

        self.horizontalLayout.setObjectName("horizontalLayout")

        self.label_9 = QtWidgets.QLabel(AdhesionFlexDlg)

        self.label_9.setObjectName("label_9")

        self.horizontalLayout.addWidget(self.label_9)

        self.afMoleculeLE = QtWidgets.QLineEdit(AdhesionFlexDlg)

        self.afMoleculeLE.setText("")

        self.afMoleculeLE.setObjectName("afMoleculeLE")

        self.horizontalLayout.addWidget(self.afMoleculeLE)

        self.afMoleculeAddPB = QtWidgets.QPushButton(AdhesionFlexDlg)

        self.afMoleculeAddPB.setObjectName("afMoleculeAddPB")

        self.horizontalLayout.addWidget(self.afMoleculeAddPB)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.label_10 = QtWidgets.QLabel(AdhesionFlexDlg)

        self.label_10.setObjectName("label_10")

        self.horizontalLayout_2.addWidget(self.label_10)

        self.bindingFormulaLE = QtWidgets.QLineEdit(AdhesionFlexDlg)

        self.bindingFormulaLE.setObjectName("bindingFormulaLE")

        self.horizontalLayout_2.addWidget(self.bindingFormulaLE)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        self.clearAFTablePB = QtWidgets.QPushButton(AdhesionFlexDlg)

        self.clearAFTablePB.setObjectName("clearAFTablePB")

        self.horizontalLayout_4.addWidget(self.clearAFTablePB)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(spacerItem)

        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_3.setObjectName("horizontalLayout_3")

        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(spacerItem1)

        self.okPB = QtWidgets.QPushButton(AdhesionFlexDlg)

        self.okPB.setObjectName("okPB")

        self.horizontalLayout_3.addWidget(self.okPB)

        self.cancelPB = QtWidgets.QPushButton(AdhesionFlexDlg)

        self.cancelPB.setObjectName("cancelPB")

        self.horizontalLayout_3.addWidget(self.cancelPB)

        self.verticalLayout_2.addLayout(self.horizontalLayout_3)



        self.retranslateUi(AdhesionFlexDlg)

        self.okPB.clicked.connect(AdhesionFlexDlg.accept)

        self.cancelPB.clicked.connect(AdhesionFlexDlg.reject)

        QtCore.QMetaObject.connectSlotsByName(AdhesionFlexDlg)



    def retranslateUi(self, AdhesionFlexDlg):

        _translate = QtCore.QCoreApplication.translate

        AdhesionFlexDlg.setWindowTitle(_translate("AdhesionFlexDlg", "Please define adhesion molecules"))

        item = self.afTable.horizontalHeaderItem(0)

        item.setText(_translate("AdhesionFlexDlg", "Adhesion Molecule"))

        self.label_9.setText(_translate("AdhesionFlexDlg", "Molecule"))

        self.afMoleculeLE.setToolTip(_translate("AdhesionFlexDlg", "Specify names of the adhesion molecules you want to use int he simulation"))

        self.afMoleculeAddPB.setText(_translate("AdhesionFlexDlg", "Add"))

        self.label_10.setText(_translate("AdhesionFlexDlg", "Binding Formula"))

        self.bindingFormulaLE.setToolTip(_translate("AdhesionFlexDlg", "This is binary function that takes atwo arguments -  Molecule1 and Molecule2. The allowed functions are those given by muParser - see http://muparser.sourceforge.net/"))

        self.bindingFormulaLE.setText(_translate("AdhesionFlexDlg", "min(Molecule1,Molecule2)"))

        self.clearAFTablePB.setText(_translate("AdhesionFlexDlg", "Clear Table"))

        self.okPB.setText(_translate("AdhesionFlexDlg", "OK"))

        self.cancelPB.setText(_translate("AdhesionFlexDlg", "Cancel"))



