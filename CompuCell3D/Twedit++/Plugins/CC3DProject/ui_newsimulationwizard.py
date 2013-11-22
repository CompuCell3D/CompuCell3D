# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\NewSimulationWizard.ui'
#
# Created: Fri Nov 22 15:22:28 2013
#      by: PyQt4 UI code generator 4.10.3
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

class Ui_NewSimulationWizard(object):
    def setupUi(self, NewSimulationWizard):
        NewSimulationWizard.setObjectName(_fromUtf8("NewSimulationWizard"))
        NewSimulationWizard.resize(615, 555)
        self.wizardPage1 = QtGui.QWizardPage()
        self.wizardPage1.setObjectName(_fromUtf8("wizardPage1"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.wizardPage1)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.label = QtGui.QLabel(self.wizardPage1)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout_3.addWidget(self.label)
        spacerItem = QtGui.QSpacerItem(20, 9, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_2 = QtGui.QLabel(self.wizardPage1)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.nameLE = QtGui.QLineEdit(self.wizardPage1)
        self.nameLE.setObjectName(_fromUtf8("nameLE"))
        self.gridLayout.addWidget(self.nameLE, 0, 1, 1, 1)
        self.label_3 = QtGui.QLabel(self.wizardPage1)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.dirLE = QtGui.QLineEdit(self.wizardPage1)
        self.dirLE.setObjectName(_fromUtf8("dirLE"))
        self.horizontalLayout_2.addWidget(self.dirLE)
        self.dirPB = QtGui.QPushButton(self.wizardPage1)
        self.dirPB.setObjectName(_fromUtf8("dirPB"))
        self.horizontalLayout_2.addWidget(self.dirPB)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 1, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout)
        spacerItem1 = QtGui.QSpacerItem(20, 9, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.groupBox = QtGui.QGroupBox(self.wizardPage1)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.xmlRB = QtGui.QRadioButton(self.groupBox)
        self.xmlRB.setObjectName(_fromUtf8("xmlRB"))
        self.verticalLayout.addWidget(self.xmlRB)
        self.pythonXMLRB = QtGui.QRadioButton(self.groupBox)
        self.pythonXMLRB.setChecked(True)
        self.pythonXMLRB.setObjectName(_fromUtf8("pythonXMLRB"))
        self.verticalLayout.addWidget(self.pythonXMLRB)
        self.pythonOnlyRB = QtGui.QRadioButton(self.groupBox)
        self.pythonOnlyRB.setObjectName(_fromUtf8("pythonOnlyRB"))
        self.verticalLayout.addWidget(self.pythonOnlyRB)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout.addWidget(self.groupBox)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        NewSimulationWizard.addPage(self.wizardPage1)
        self.wizardPage_9 = QtGui.QWizardPage()
        self.wizardPage_9.setObjectName(_fromUtf8("wizardPage_9"))
        self.gridLayout_7 = QtGui.QGridLayout(self.wizardPage_9)
        self.gridLayout_7.setObjectName(_fromUtf8("gridLayout_7"))
        self.label_27 = QtGui.QLabel(self.wizardPage_9)
        self.label_27.setObjectName(_fromUtf8("label_27"))
        self.gridLayout_7.addWidget(self.label_27, 0, 0, 1, 1)
        self.verticalLayout_19 = QtGui.QVBoxLayout()
        self.verticalLayout_19.setObjectName(_fromUtf8("verticalLayout_19"))
        self.groupBox_7 = QtGui.QGroupBox(self.wizardPage_9)
        self.groupBox_7.setObjectName(_fromUtf8("groupBox_7"))
        self.horizontalLayout_25 = QtGui.QHBoxLayout(self.groupBox_7)
        self.horizontalLayout_25.setObjectName(_fromUtf8("horizontalLayout_25"))
        self.horizontalLayout_24 = QtGui.QHBoxLayout()
        self.horizontalLayout_24.setObjectName(_fromUtf8("horizontalLayout_24"))
        self.label_28 = QtGui.QLabel(self.groupBox_7)
        self.label_28.setObjectName(_fromUtf8("label_28"))
        self.horizontalLayout_24.addWidget(self.label_28)
        self.xDimSB = QtGui.QSpinBox(self.groupBox_7)
        self.xDimSB.setMinimum(1)
        self.xDimSB.setMaximum(10000000)
        self.xDimSB.setProperty("value", 100)
        self.xDimSB.setObjectName(_fromUtf8("xDimSB"))
        self.horizontalLayout_24.addWidget(self.xDimSB)
        self.label_29 = QtGui.QLabel(self.groupBox_7)
        self.label_29.setObjectName(_fromUtf8("label_29"))
        self.horizontalLayout_24.addWidget(self.label_29)
        self.yDimSB = QtGui.QSpinBox(self.groupBox_7)
        self.yDimSB.setMinimum(1)
        self.yDimSB.setMaximum(10000000)
        self.yDimSB.setProperty("value", 100)
        self.yDimSB.setObjectName(_fromUtf8("yDimSB"))
        self.horizontalLayout_24.addWidget(self.yDimSB)
        self.label_30 = QtGui.QLabel(self.groupBox_7)
        self.label_30.setObjectName(_fromUtf8("label_30"))
        self.horizontalLayout_24.addWidget(self.label_30)
        self.zDimSB = QtGui.QSpinBox(self.groupBox_7)
        self.zDimSB.setMinimum(1)
        self.zDimSB.setMaximum(10000000)
        self.zDimSB.setObjectName(_fromUtf8("zDimSB"))
        self.horizontalLayout_24.addWidget(self.zDimSB)
        self.horizontalLayout_25.addLayout(self.horizontalLayout_24)
        self.verticalLayout_19.addWidget(self.groupBox_7)
        self.gridLayout_6 = QtGui.QGridLayout()
        self.gridLayout_6.setObjectName(_fromUtf8("gridLayout_6"))
        self.label_31 = QtGui.QLabel(self.wizardPage_9)
        self.label_31.setObjectName(_fromUtf8("label_31"))
        self.gridLayout_6.addWidget(self.label_31, 1, 0, 1, 1)
        self.membraneFluctuationsLE = QtGui.QLineEdit(self.wizardPage_9)
        self.membraneFluctuationsLE.setObjectName(_fromUtf8("membraneFluctuationsLE"))
        self.gridLayout_6.addWidget(self.membraneFluctuationsLE, 1, 1, 1, 1)
        self.label_32 = QtGui.QLabel(self.wizardPage_9)
        self.label_32.setObjectName(_fromUtf8("label_32"))
        self.gridLayout_6.addWidget(self.label_32, 2, 0, 1, 1)
        self.neighborOrderSB = QtGui.QSpinBox(self.wizardPage_9)
        self.neighborOrderSB.setMinimum(1)
        self.neighborOrderSB.setMaximum(10)
        self.neighborOrderSB.setObjectName(_fromUtf8("neighborOrderSB"))
        self.gridLayout_6.addWidget(self.neighborOrderSB, 2, 1, 1, 1)
        self.label_33 = QtGui.QLabel(self.wizardPage_9)
        self.label_33.setObjectName(_fromUtf8("label_33"))
        self.gridLayout_6.addWidget(self.label_33, 3, 0, 1, 1)
        self.mcsSB = QtGui.QSpinBox(self.wizardPage_9)
        self.mcsSB.setMaximum(1000000000)
        self.mcsSB.setProperty("value", 1000)
        self.mcsSB.setObjectName(_fromUtf8("mcsSB"))
        self.gridLayout_6.addWidget(self.mcsSB, 3, 1, 1, 1)
        self.latticeTypeCB = QtGui.QComboBox(self.wizardPage_9)
        self.latticeTypeCB.setObjectName(_fromUtf8("latticeTypeCB"))
        self.latticeTypeCB.addItem(_fromUtf8(""))
        self.latticeTypeCB.addItem(_fromUtf8(""))
        self.gridLayout_6.addWidget(self.latticeTypeCB, 0, 1, 1, 1)
        self.label_44 = QtGui.QLabel(self.wizardPage_9)
        self.label_44.setObjectName(_fromUtf8("label_44"))
        self.gridLayout_6.addWidget(self.label_44, 0, 0, 1, 1)
        self.verticalLayout_19.addLayout(self.gridLayout_6)
        self.line_3 = QtGui.QFrame(self.wizardPage_9)
        self.line_3.setFrameShape(QtGui.QFrame.HLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.verticalLayout_19.addWidget(self.line_3)
        self.groupBox_8 = QtGui.QGroupBox(self.wizardPage_9)
        self.groupBox_8.setObjectName(_fromUtf8("groupBox_8"))
        self.verticalLayout_18 = QtGui.QVBoxLayout(self.groupBox_8)
        self.verticalLayout_18.setObjectName(_fromUtf8("verticalLayout_18"))
        self.gridLayout_5 = QtGui.QGridLayout()
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.uniformRB = QtGui.QRadioButton(self.groupBox_8)
        self.uniformRB.setChecked(True)
        self.uniformRB.setObjectName(_fromUtf8("uniformRB"))
        self.gridLayout_5.addWidget(self.uniformRB, 0, 0, 1, 1)
        self.blobRB = QtGui.QRadioButton(self.groupBox_8)
        self.blobRB.setObjectName(_fromUtf8("blobRB"))
        self.gridLayout_5.addWidget(self.blobRB, 0, 1, 1, 1)
        self.piffRB = QtGui.QRadioButton(self.groupBox_8)
        self.piffRB.setObjectName(_fromUtf8("piffRB"))
        self.gridLayout_5.addWidget(self.piffRB, 0, 2, 1, 1)
        self.piffLE = QtGui.QLineEdit(self.groupBox_8)
        self.piffLE.setObjectName(_fromUtf8("piffLE"))
        self.gridLayout_5.addWidget(self.piffLE, 1, 1, 1, 2)
        self.piffPB = QtGui.QPushButton(self.groupBox_8)
        self.piffPB.setObjectName(_fromUtf8("piffPB"))
        self.gridLayout_5.addWidget(self.piffPB, 1, 0, 1, 1)
        self.verticalLayout_18.addLayout(self.gridLayout_5)
        self.verticalLayout_19.addWidget(self.groupBox_8)
        self.gridLayout_7.addLayout(self.verticalLayout_19, 1, 0, 1, 1)
        spacerItem3 = QtGui.QSpacerItem(238, 54, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem3, 1, 1, 1, 1)
        spacerItem4 = QtGui.QSpacerItem(20, 155, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_7.addItem(spacerItem4, 2, 0, 1, 1)
        NewSimulationWizard.addPage(self.wizardPage_9)
        self.wizardPage2 = QtGui.QWizardPage()
        self.wizardPage2.setObjectName(_fromUtf8("wizardPage2"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.wizardPage2)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.label_5 = QtGui.QLabel(self.wizardPage2)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.verticalLayout_4.addWidget(self.label_5)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.cellTypeTable = QtGui.QTableWidget(self.wizardPage2)
        self.cellTypeTable.setEnabled(True)
        self.cellTypeTable.setBaseSize(QtCore.QSize(256, 171))
        self.cellTypeTable.setObjectName(_fromUtf8("cellTypeTable"))
        self.cellTypeTable.setColumnCount(2)
        self.cellTypeTable.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.cellTypeTable.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.cellTypeTable.setHorizontalHeaderItem(1, item)
        self.horizontalLayout_4.addWidget(self.cellTypeTable)
        self.clearCellTypeTablePB = QtGui.QPushButton(self.wizardPage2)
        self.clearCellTypeTablePB.setObjectName(_fromUtf8("clearCellTypeTablePB"))
        self.horizontalLayout_4.addWidget(self.clearCellTypeTablePB)
        spacerItem5 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem5)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_4 = QtGui.QLabel(self.wizardPage2)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout_3.addWidget(self.label_4)
        self.cellTypeLE = QtGui.QLineEdit(self.wizardPage2)
        self.cellTypeLE.setObjectName(_fromUtf8("cellTypeLE"))
        self.horizontalLayout_3.addWidget(self.cellTypeLE)
        self.freezeCHB = QtGui.QCheckBox(self.wizardPage2)
        self.freezeCHB.setObjectName(_fromUtf8("freezeCHB"))
        self.horizontalLayout_3.addWidget(self.freezeCHB)
        self.cellTypeAddPB = QtGui.QPushButton(self.wizardPage2)
        self.cellTypeAddPB.setObjectName(_fromUtf8("cellTypeAddPB"))
        self.horizontalLayout_3.addWidget(self.cellTypeAddPB)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        NewSimulationWizard.addPage(self.wizardPage2)
        self.wizardPage_7 = QtGui.QWizardPage()
        self.wizardPage_7.setObjectName(_fromUtf8("wizardPage_7"))
        self.verticalLayout_11 = QtGui.QVBoxLayout(self.wizardPage_7)
        self.verticalLayout_11.setObjectName(_fromUtf8("verticalLayout_11"))
        self.label_17 = QtGui.QLabel(self.wizardPage_7)
        self.label_17.setObjectName(_fromUtf8("label_17"))
        self.verticalLayout_11.addWidget(self.label_17)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.fieldTable = QtGui.QTableWidget(self.wizardPage_7)
        self.fieldTable.setMinimumSize(QtCore.QSize(300, 0))
        self.fieldTable.setBaseSize(QtCore.QSize(300, 0))
        self.fieldTable.setObjectName(_fromUtf8("fieldTable"))
        self.fieldTable.setColumnCount(2)
        self.fieldTable.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.fieldTable.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.fieldTable.setHorizontalHeaderItem(1, item)
        self.horizontalLayout_5.addWidget(self.fieldTable)
        self.clearFieldTablePB = QtGui.QPushButton(self.wizardPage_7)
        self.clearFieldTablePB.setObjectName(_fromUtf8("clearFieldTablePB"))
        self.horizontalLayout_5.addWidget(self.clearFieldTablePB)
        spacerItem6 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem6)
        self.verticalLayout_11.addLayout(self.horizontalLayout_5)
        self.pdeSolverCallerCHB = QtGui.QCheckBox(self.wizardPage_7)
        self.pdeSolverCallerCHB.setObjectName(_fromUtf8("pdeSolverCallerCHB"))
        self.verticalLayout_11.addWidget(self.pdeSolverCallerCHB)
        self.horizontalLayout_15 = QtGui.QHBoxLayout()
        self.horizontalLayout_15.setObjectName(_fromUtf8("horizontalLayout_15"))
        self.label_18 = QtGui.QLabel(self.wizardPage_7)
        self.label_18.setObjectName(_fromUtf8("label_18"))
        self.horizontalLayout_15.addWidget(self.label_18)
        self.fieldNameLE = QtGui.QLineEdit(self.wizardPage_7)
        self.fieldNameLE.setMinimumSize(QtCore.QSize(120, 0))
        self.fieldNameLE.setObjectName(_fromUtf8("fieldNameLE"))
        self.horizontalLayout_15.addWidget(self.fieldNameLE)
        self.label_19 = QtGui.QLabel(self.wizardPage_7)
        self.label_19.setObjectName(_fromUtf8("label_19"))
        self.horizontalLayout_15.addWidget(self.label_19)
        self.solverCB = QtGui.QComboBox(self.wizardPage_7)
        self.solverCB.setObjectName(_fromUtf8("solverCB"))
        self.solverCB.addItem(_fromUtf8(""))
        self.solverCB.addItem(_fromUtf8(""))
        self.solverCB.addItem(_fromUtf8(""))
        self.solverCB.addItem(_fromUtf8(""))
        self.solverCB.addItem(_fromUtf8(""))
        self.horizontalLayout_15.addWidget(self.solverCB)
        self.fieldAddPB = QtGui.QPushButton(self.wizardPage_7)
        self.fieldAddPB.setObjectName(_fromUtf8("fieldAddPB"))
        self.horizontalLayout_15.addWidget(self.fieldAddPB)
        self.verticalLayout_11.addLayout(self.horizontalLayout_15)
        NewSimulationWizard.addPage(self.wizardPage_7)
        self.wizardPage = QtGui.QWizardPage()
        self.wizardPage.setObjectName(_fromUtf8("wizardPage"))
        self.gridLayout_8 = QtGui.QGridLayout(self.wizardPage)
        self.gridLayout_8.setObjectName(_fromUtf8("gridLayout_8"))
        self.label_6 = QtGui.QLabel(self.wizardPage)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout_8.addWidget(self.label_6, 0, 0, 1, 2)
        self.groupBox_9 = QtGui.QGroupBox(self.wizardPage)
        self.groupBox_9.setObjectName(_fromUtf8("groupBox_9"))
        self.verticalLayout_6 = QtGui.QVBoxLayout(self.groupBox_9)
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.label_34 = QtGui.QLabel(self.groupBox_9)
        self.label_34.setObjectName(_fromUtf8("label_34"))
        self.verticalLayout_6.addWidget(self.label_34)
        self.contactCHB = QtGui.QCheckBox(self.groupBox_9)
        self.contactCHB.setObjectName(_fromUtf8("contactCHB"))
        self.verticalLayout_6.addWidget(self.contactCHB)
        self.internalContactCB = QtGui.QCheckBox(self.groupBox_9)
        self.internalContactCB.setObjectName(_fromUtf8("internalContactCB"))
        self.verticalLayout_6.addWidget(self.internalContactCB)
        self.adhesionFlexCHB = QtGui.QCheckBox(self.groupBox_9)
        self.adhesionFlexCHB.setObjectName(_fromUtf8("adhesionFlexCHB"))
        self.verticalLayout_6.addWidget(self.adhesionFlexCHB)
        self.contactLocalProductCHB = QtGui.QCheckBox(self.groupBox_9)
        self.contactLocalProductCHB.setObjectName(_fromUtf8("contactLocalProductCHB"))
        self.verticalLayout_6.addWidget(self.contactLocalProductCHB)
        self.compartmentCHB = QtGui.QCheckBox(self.groupBox_9)
        self.compartmentCHB.setObjectName(_fromUtf8("compartmentCHB"))
        self.verticalLayout_6.addWidget(self.compartmentCHB)
        self.fppCHB = QtGui.QCheckBox(self.groupBox_9)
        self.fppCHB.setObjectName(_fromUtf8("fppCHB"))
        self.verticalLayout_6.addWidget(self.fppCHB)
        self.elasticityCHB = QtGui.QCheckBox(self.groupBox_9)
        self.elasticityCHB.setObjectName(_fromUtf8("elasticityCHB"))
        self.verticalLayout_6.addWidget(self.elasticityCHB)
        self.contactMultiCadCHB = QtGui.QCheckBox(self.groupBox_9)
        self.contactMultiCadCHB.setCheckable(True)
        self.contactMultiCadCHB.setObjectName(_fromUtf8("contactMultiCadCHB"))
        self.verticalLayout_6.addWidget(self.contactMultiCadCHB)
        self.label_35 = QtGui.QLabel(self.groupBox_9)
        self.label_35.setObjectName(_fromUtf8("label_35"))
        self.verticalLayout_6.addWidget(self.label_35)
        self.chemotaxisCHB = QtGui.QCheckBox(self.groupBox_9)
        self.chemotaxisCHB.setObjectName(_fromUtf8("chemotaxisCHB"))
        self.verticalLayout_6.addWidget(self.chemotaxisCHB)
        self.label_36 = QtGui.QLabel(self.groupBox_9)
        self.label_36.setObjectName(_fromUtf8("label_36"))
        self.verticalLayout_6.addWidget(self.label_36)
        self.secretionCHB = QtGui.QCheckBox(self.groupBox_9)
        self.secretionCHB.setObjectName(_fromUtf8("secretionCHB"))
        self.verticalLayout_6.addWidget(self.secretionCHB)
        self.label_37 = QtGui.QLabel(self.groupBox_9)
        self.label_37.setObjectName(_fromUtf8("label_37"))
        self.verticalLayout_6.addWidget(self.label_37)
        self.growthCHB = QtGui.QCheckBox(self.groupBox_9)
        self.growthCHB.setObjectName(_fromUtf8("growthCHB"))
        self.verticalLayout_6.addWidget(self.growthCHB)
        self.label_38 = QtGui.QLabel(self.groupBox_9)
        self.label_38.setObjectName(_fromUtf8("label_38"))
        self.verticalLayout_6.addWidget(self.label_38)
        self.mitosisCHB = QtGui.QCheckBox(self.groupBox_9)
        self.mitosisCHB.setObjectName(_fromUtf8("mitosisCHB"))
        self.verticalLayout_6.addWidget(self.mitosisCHB)
        self.label_39 = QtGui.QLabel(self.groupBox_9)
        self.label_39.setObjectName(_fromUtf8("label_39"))
        self.verticalLayout_6.addWidget(self.label_39)
        self.deathCHB = QtGui.QCheckBox(self.groupBox_9)
        self.deathCHB.setObjectName(_fromUtf8("deathCHB"))
        self.verticalLayout_6.addWidget(self.deathCHB)
        self.gridLayout_8.addWidget(self.groupBox_9, 1, 0, 2, 1)
        self.groupBox_2 = QtGui.QGroupBox(self.wizardPage)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.verticalLayout_13 = QtGui.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_13.setObjectName(_fromUtf8("verticalLayout_13"))
        self.label_40 = QtGui.QLabel(self.groupBox_2)
        self.label_40.setObjectName(_fromUtf8("label_40"))
        self.verticalLayout_13.addWidget(self.label_40)
        self.volumeFlexCHB = QtGui.QCheckBox(self.groupBox_2)
        self.volumeFlexCHB.setObjectName(_fromUtf8("volumeFlexCHB"))
        self.verticalLayout_13.addWidget(self.volumeFlexCHB)
        self.volumeLocalFlexCHB = QtGui.QCheckBox(self.groupBox_2)
        self.volumeLocalFlexCHB.setObjectName(_fromUtf8("volumeLocalFlexCHB"))
        self.verticalLayout_13.addWidget(self.volumeLocalFlexCHB)
        self.label_41 = QtGui.QLabel(self.groupBox_2)
        self.label_41.setObjectName(_fromUtf8("label_41"))
        self.verticalLayout_13.addWidget(self.label_41)
        self.surfaceFlexCHB = QtGui.QCheckBox(self.groupBox_2)
        self.surfaceFlexCHB.setObjectName(_fromUtf8("surfaceFlexCHB"))
        self.verticalLayout_13.addWidget(self.surfaceFlexCHB)
        self.surfaceLocalFlexCHB = QtGui.QCheckBox(self.groupBox_2)
        self.surfaceLocalFlexCHB.setObjectName(_fromUtf8("surfaceLocalFlexCHB"))
        self.verticalLayout_13.addWidget(self.surfaceLocalFlexCHB)
        self.label_42 = QtGui.QLabel(self.groupBox_2)
        self.label_42.setObjectName(_fromUtf8("label_42"))
        self.verticalLayout_13.addWidget(self.label_42)
        self.extPotCHB = QtGui.QCheckBox(self.groupBox_2)
        self.extPotCHB.setObjectName(_fromUtf8("extPotCHB"))
        self.verticalLayout_13.addWidget(self.extPotCHB)
        self.extPotLocalFlexCHB = QtGui.QCheckBox(self.groupBox_2)
        self.extPotLocalFlexCHB.setObjectName(_fromUtf8("extPotLocalFlexCHB"))
        self.verticalLayout_13.addWidget(self.extPotLocalFlexCHB)
        self.label_43 = QtGui.QLabel(self.groupBox_2)
        self.label_43.setObjectName(_fromUtf8("label_43"))
        self.verticalLayout_13.addWidget(self.label_43)
        self.connectGlobalCHB = QtGui.QCheckBox(self.groupBox_2)
        self.connectGlobalCHB.setObjectName(_fromUtf8("connectGlobalCHB"))
        self.verticalLayout_13.addWidget(self.connectGlobalCHB)
        self.connectGlobalByIdCHB = QtGui.QCheckBox(self.groupBox_2)
        self.connectGlobalByIdCHB.setObjectName(_fromUtf8("connectGlobalByIdCHB"))
        self.verticalLayout_13.addWidget(self.connectGlobalByIdCHB)
        self.connect2DCHB = QtGui.QCheckBox(self.groupBox_2)
        self.connect2DCHB.setObjectName(_fromUtf8("connect2DCHB"))
        self.verticalLayout_13.addWidget(self.connect2DCHB)
        self.label_45 = QtGui.QLabel(self.groupBox_2)
        self.label_45.setObjectName(_fromUtf8("label_45"))
        self.verticalLayout_13.addWidget(self.label_45)
        self.lengthConstraintCHB = QtGui.QCheckBox(self.groupBox_2)
        self.lengthConstraintCHB.setObjectName(_fromUtf8("lengthConstraintCHB"))
        self.verticalLayout_13.addWidget(self.lengthConstraintCHB)
        self.lengthConstraintLocalFlexCHB = QtGui.QCheckBox(self.groupBox_2)
        self.lengthConstraintLocalFlexCHB.setObjectName(_fromUtf8("lengthConstraintLocalFlexCHB"))
        self.verticalLayout_13.addWidget(self.lengthConstraintLocalFlexCHB)
        self.gridLayout_8.addWidget(self.groupBox_2, 1, 1, 2, 1)
        self.groupBox_4 = QtGui.QGroupBox(self.wizardPage)
        self.groupBox_4.setObjectName(_fromUtf8("groupBox_4"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.comCHB = QtGui.QCheckBox(self.groupBox_4)
        self.comCHB.setChecked(True)
        self.comCHB.setObjectName(_fromUtf8("comCHB"))
        self.verticalLayout_5.addWidget(self.comCHB)
        self.neighborCHB = QtGui.QCheckBox(self.groupBox_4)
        self.neighborCHB.setObjectName(_fromUtf8("neighborCHB"))
        self.verticalLayout_5.addWidget(self.neighborCHB)
        self.momentOfInertiaCHB = QtGui.QCheckBox(self.groupBox_4)
        self.momentOfInertiaCHB.setObjectName(_fromUtf8("momentOfInertiaCHB"))
        self.verticalLayout_5.addWidget(self.momentOfInertiaCHB)
        self.pixelTrackerCHB = QtGui.QCheckBox(self.groupBox_4)
        self.pixelTrackerCHB.setObjectName(_fromUtf8("pixelTrackerCHB"))
        self.verticalLayout_5.addWidget(self.pixelTrackerCHB)
        self.boundaryPixelTrackerCHB = QtGui.QCheckBox(self.groupBox_4)
        self.boundaryPixelTrackerCHB.setObjectName(_fromUtf8("boundaryPixelTrackerCHB"))
        self.verticalLayout_5.addWidget(self.boundaryPixelTrackerCHB)
        self.gridLayout_8.addWidget(self.groupBox_4, 1, 2, 1, 1)
        self.groupBox_10 = QtGui.QGroupBox(self.wizardPage)
        self.groupBox_10.setObjectName(_fromUtf8("groupBox_10"))
        self.verticalLayout_12 = QtGui.QVBoxLayout(self.groupBox_10)
        self.verticalLayout_12.setObjectName(_fromUtf8("verticalLayout_12"))
        self.boxWatcherCHB = QtGui.QCheckBox(self.groupBox_10)
        self.boxWatcherCHB.setObjectName(_fromUtf8("boxWatcherCHB"))
        self.verticalLayout_12.addWidget(self.boxWatcherCHB)
        self.pifDumperCHB = QtGui.QCheckBox(self.groupBox_10)
        self.pifDumperCHB.setObjectName(_fromUtf8("pifDumperCHB"))
        self.verticalLayout_12.addWidget(self.pifDumperCHB)
        self.gridLayout_8.addWidget(self.groupBox_10, 2, 2, 1, 1)
        NewSimulationWizard.addPage(self.wizardPage)
        self.wizardPage_8 = QtGui.QWizardPage()
        self.wizardPage_8.setObjectName(_fromUtf8("wizardPage_8"))
        self.verticalLayout_17 = QtGui.QVBoxLayout(self.wizardPage_8)
        self.verticalLayout_17.setObjectName(_fromUtf8("verticalLayout_17"))
        self.label_24 = QtGui.QLabel(self.wizardPage_8)
        self.label_24.setObjectName(_fromUtf8("label_24"))
        self.verticalLayout_17.addWidget(self.label_24)
        self.secretionTable = QtGui.QTableWidget(self.wizardPage_8)
        self.secretionTable.setBaseSize(QtCore.QSize(580, 0))
        self.secretionTable.setObjectName(_fromUtf8("secretionTable"))
        self.secretionTable.setColumnCount(5)
        self.secretionTable.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.secretionTable.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.secretionTable.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        self.secretionTable.setHorizontalHeaderItem(2, item)
        item = QtGui.QTableWidgetItem()
        self.secretionTable.setHorizontalHeaderItem(3, item)
        item = QtGui.QTableWidgetItem()
        self.secretionTable.setHorizontalHeaderItem(4, item)
        self.verticalLayout_17.addWidget(self.secretionTable)
        self.gridLayout_4 = QtGui.QGridLayout()
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.groupBox_6 = QtGui.QGroupBox(self.wizardPage_8)
        self.groupBox_6.setObjectName(_fromUtf8("groupBox_6"))
        self.verticalLayout_16 = QtGui.QVBoxLayout(self.groupBox_6)
        self.verticalLayout_16.setObjectName(_fromUtf8("verticalLayout_16"))
        self.horizontalLayout_20 = QtGui.QHBoxLayout()
        self.horizontalLayout_20.setObjectName(_fromUtf8("horizontalLayout_20"))
        self.secrUniformRB = QtGui.QRadioButton(self.groupBox_6)
        self.secrUniformRB.setChecked(True)
        self.secrUniformRB.setObjectName(_fromUtf8("secrUniformRB"))
        self.horizontalLayout_20.addWidget(self.secrUniformRB)
        self.secrOnContactRB = QtGui.QRadioButton(self.groupBox_6)
        self.secrOnContactRB.setObjectName(_fromUtf8("secrOnContactRB"))
        self.horizontalLayout_20.addWidget(self.secrOnContactRB)
        self.secrConstConcRB = QtGui.QRadioButton(self.groupBox_6)
        self.secrConstConcRB.setObjectName(_fromUtf8("secrConstConcRB"))
        self.horizontalLayout_20.addWidget(self.secrConstConcRB)
        self.verticalLayout_16.addLayout(self.horizontalLayout_20)
        self.gridLayout_4.addWidget(self.groupBox_6, 0, 0, 1, 2)
        self.horizontalLayout_21 = QtGui.QHBoxLayout()
        self.horizontalLayout_21.setObjectName(_fromUtf8("horizontalLayout_21"))
        self.label_25 = QtGui.QLabel(self.wizardPage_8)
        self.label_25.setObjectName(_fromUtf8("label_25"))
        self.horizontalLayout_21.addWidget(self.label_25)
        self.secrFieldCB = QtGui.QComboBox(self.wizardPage_8)
        self.secrFieldCB.setObjectName(_fromUtf8("secrFieldCB"))
        self.horizontalLayout_21.addWidget(self.secrFieldCB)
        self.label_26 = QtGui.QLabel(self.wizardPage_8)
        self.label_26.setObjectName(_fromUtf8("label_26"))
        self.horizontalLayout_21.addWidget(self.label_26)
        self.secrCellTypeCB = QtGui.QComboBox(self.wizardPage_8)
        self.secrCellTypeCB.setObjectName(_fromUtf8("secrCellTypeCB"))
        self.horizontalLayout_21.addWidget(self.secrCellTypeCB)
        self.secrRateLB = QtGui.QLabel(self.wizardPage_8)
        self.secrRateLB.setObjectName(_fromUtf8("secrRateLB"))
        self.horizontalLayout_21.addWidget(self.secrRateLB)
        self.secrRateLE = QtGui.QLineEdit(self.wizardPage_8)
        self.secrRateLE.setObjectName(_fromUtf8("secrRateLE"))
        self.horizontalLayout_21.addWidget(self.secrRateLE)
        self.gridLayout_4.addLayout(self.horizontalLayout_21, 1, 0, 1, 2)
        spacerItem7 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem7, 1, 2, 1, 1)
        self.horizontalLayout_22 = QtGui.QHBoxLayout()
        self.horizontalLayout_22.setObjectName(_fromUtf8("horizontalLayout_22"))
        self.secrAddOnContactPB = QtGui.QPushButton(self.wizardPage_8)
        self.secrAddOnContactPB.setObjectName(_fromUtf8("secrAddOnContactPB"))
        self.horizontalLayout_22.addWidget(self.secrAddOnContactPB)
        self.secrOnContactCellTypeCB = QtGui.QComboBox(self.wizardPage_8)
        self.secrOnContactCellTypeCB.setObjectName(_fromUtf8("secrOnContactCellTypeCB"))
        self.horizontalLayout_22.addWidget(self.secrOnContactCellTypeCB)
        self.secrOnContactLE = QtGui.QLineEdit(self.wizardPage_8)
        self.secrOnContactLE.setObjectName(_fromUtf8("secrOnContactLE"))
        self.horizontalLayout_22.addWidget(self.secrOnContactLE)
        self.gridLayout_4.addLayout(self.horizontalLayout_22, 2, 0, 1, 1)
        spacerItem8 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem8, 2, 1, 1, 1)
        self.verticalLayout_17.addLayout(self.gridLayout_4)
        self.line_2 = QtGui.QFrame(self.wizardPage_8)
        self.line_2.setFrameShape(QtGui.QFrame.HLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.verticalLayout_17.addWidget(self.line_2)
        self.horizontalLayout_23 = QtGui.QHBoxLayout()
        self.horizontalLayout_23.setObjectName(_fromUtf8("horizontalLayout_23"))
        self.secrAddRowPB = QtGui.QPushButton(self.wizardPage_8)
        self.secrAddRowPB.setObjectName(_fromUtf8("secrAddRowPB"))
        self.horizontalLayout_23.addWidget(self.secrAddRowPB)
        spacerItem9 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_23.addItem(spacerItem9)
        self.secrRemoveRowsPB = QtGui.QPushButton(self.wizardPage_8)
        self.secrRemoveRowsPB.setObjectName(_fromUtf8("secrRemoveRowsPB"))
        self.horizontalLayout_23.addWidget(self.secrRemoveRowsPB)
        self.secrClearTablePB = QtGui.QPushButton(self.wizardPage_8)
        self.secrClearTablePB.setObjectName(_fromUtf8("secrClearTablePB"))
        self.horizontalLayout_23.addWidget(self.secrClearTablePB)
        self.verticalLayout_17.addLayout(self.horizontalLayout_23)
        NewSimulationWizard.addPage(self.wizardPage_8)
        self.wizardPage_6 = QtGui.QWizardPage()
        self.wizardPage_6.setObjectName(_fromUtf8("wizardPage_6"))
        self.verticalLayout_15 = QtGui.QVBoxLayout(self.wizardPage_6)
        self.verticalLayout_15.setObjectName(_fromUtf8("verticalLayout_15"))
        self.label_16 = QtGui.QLabel(self.wizardPage_6)
        self.label_16.setObjectName(_fromUtf8("label_16"))
        self.verticalLayout_15.addWidget(self.label_16)
        self.chamotaxisTable = QtGui.QTableWidget(self.wizardPage_6)
        self.chamotaxisTable.setObjectName(_fromUtf8("chamotaxisTable"))
        self.chamotaxisTable.setColumnCount(6)
        self.chamotaxisTable.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.chamotaxisTable.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.chamotaxisTable.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        self.chamotaxisTable.setHorizontalHeaderItem(2, item)
        item = QtGui.QTableWidgetItem()
        self.chamotaxisTable.setHorizontalHeaderItem(3, item)
        item = QtGui.QTableWidgetItem()
        self.chamotaxisTable.setHorizontalHeaderItem(4, item)
        item = QtGui.QTableWidgetItem()
        self.chamotaxisTable.setHorizontalHeaderItem(5, item)
        self.verticalLayout_15.addWidget(self.chamotaxisTable)
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.groupBox_5 = QtGui.QGroupBox(self.wizardPage_6)
        self.groupBox_5.setObjectName(_fromUtf8("groupBox_5"))
        self.verticalLayout_14 = QtGui.QVBoxLayout(self.groupBox_5)
        self.verticalLayout_14.setObjectName(_fromUtf8("verticalLayout_14"))
        self.horizontalLayout_17 = QtGui.QHBoxLayout()
        self.horizontalLayout_17.setObjectName(_fromUtf8("horizontalLayout_17"))
        self.chemRegRB = QtGui.QRadioButton(self.groupBox_5)
        self.chemRegRB.setChecked(True)
        self.chemRegRB.setObjectName(_fromUtf8("chemRegRB"))
        self.horizontalLayout_17.addWidget(self.chemRegRB)
        self.chemSatRB = QtGui.QRadioButton(self.groupBox_5)
        self.chemSatRB.setObjectName(_fromUtf8("chemSatRB"))
        self.horizontalLayout_17.addWidget(self.chemSatRB)
        self.chemSatLinRB = QtGui.QRadioButton(self.groupBox_5)
        self.chemSatLinRB.setObjectName(_fromUtf8("chemSatLinRB"))
        self.horizontalLayout_17.addWidget(self.chemSatLinRB)
        self.verticalLayout_14.addLayout(self.horizontalLayout_17)
        self.gridLayout_3.addWidget(self.groupBox_5, 0, 0, 1, 1)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.label_20 = QtGui.QLabel(self.wizardPage_6)
        self.label_20.setObjectName(_fromUtf8("label_20"))
        self.gridLayout_2.addWidget(self.label_20, 0, 0, 1, 1)
        self.chemFieldCB = QtGui.QComboBox(self.wizardPage_6)
        self.chemFieldCB.setObjectName(_fromUtf8("chemFieldCB"))
        self.gridLayout_2.addWidget(self.chemFieldCB, 0, 1, 1, 1)
        self.label_21 = QtGui.QLabel(self.wizardPage_6)
        self.label_21.setObjectName(_fromUtf8("label_21"))
        self.gridLayout_2.addWidget(self.label_21, 0, 2, 1, 1)
        self.chemCellTypeCB = QtGui.QComboBox(self.wizardPage_6)
        self.chemCellTypeCB.setObjectName(_fromUtf8("chemCellTypeCB"))
        self.gridLayout_2.addWidget(self.chemCellTypeCB, 0, 3, 1, 1)
        self.label_22 = QtGui.QLabel(self.wizardPage_6)
        self.label_22.setObjectName(_fromUtf8("label_22"))
        self.gridLayout_2.addWidget(self.label_22, 1, 0, 1, 1)
        self.lambdaChemLE = QtGui.QLineEdit(self.wizardPage_6)
        self.lambdaChemLE.setObjectName(_fromUtf8("lambdaChemLE"))
        self.gridLayout_2.addWidget(self.lambdaChemLE, 1, 1, 1, 1)
        self.satCoefLB = QtGui.QLabel(self.wizardPage_6)
        self.satCoefLB.setObjectName(_fromUtf8("satCoefLB"))
        self.gridLayout_2.addWidget(self.satCoefLB, 1, 2, 1, 1)
        self.satChemLE = QtGui.QLineEdit(self.wizardPage_6)
        self.satChemLE.setObjectName(_fromUtf8("satChemLE"))
        self.gridLayout_2.addWidget(self.satChemLE, 1, 3, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 1, 0, 1, 1)
        spacerItem10 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem10, 1, 1, 1, 1)
        self.horizontalLayout_18 = QtGui.QHBoxLayout()
        self.horizontalLayout_18.setObjectName(_fromUtf8("horizontalLayout_18"))
        self.chemotaxTowardsPB = QtGui.QPushButton(self.wizardPage_6)
        self.chemotaxTowardsPB.setObjectName(_fromUtf8("chemotaxTowardsPB"))
        self.horizontalLayout_18.addWidget(self.chemotaxTowardsPB)
        self.label_23 = QtGui.QLabel(self.wizardPage_6)
        self.label_23.setObjectName(_fromUtf8("label_23"))
        self.horizontalLayout_18.addWidget(self.label_23)
        self.chemTowardsCellTypeCB = QtGui.QComboBox(self.wizardPage_6)
        self.chemTowardsCellTypeCB.setObjectName(_fromUtf8("chemTowardsCellTypeCB"))
        self.horizontalLayout_18.addWidget(self.chemTowardsCellTypeCB)
        self.chemotaxTowardsLE = QtGui.QLineEdit(self.wizardPage_6)
        self.chemotaxTowardsLE.setObjectName(_fromUtf8("chemotaxTowardsLE"))
        self.horizontalLayout_18.addWidget(self.chemotaxTowardsLE)
        self.gridLayout_3.addLayout(self.horizontalLayout_18, 2, 0, 1, 1)
        spacerItem11 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem11, 2, 1, 1, 1)
        self.verticalLayout_15.addLayout(self.gridLayout_3)
        self.line = QtGui.QFrame(self.wizardPage_6)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.verticalLayout_15.addWidget(self.line)
        self.horizontalLayout_19 = QtGui.QHBoxLayout()
        self.horizontalLayout_19.setObjectName(_fromUtf8("horizontalLayout_19"))
        self.chemotaxisAddRowPB = QtGui.QPushButton(self.wizardPage_6)
        self.chemotaxisAddRowPB.setObjectName(_fromUtf8("chemotaxisAddRowPB"))
        self.horizontalLayout_19.addWidget(self.chemotaxisAddRowPB)
        spacerItem12 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_19.addItem(spacerItem12)
        self.chemotaxisRemoveRowsPB = QtGui.QPushButton(self.wizardPage_6)
        self.chemotaxisRemoveRowsPB.setObjectName(_fromUtf8("chemotaxisRemoveRowsPB"))
        self.horizontalLayout_19.addWidget(self.chemotaxisRemoveRowsPB)
        self.chemotaxisClearTablePB = QtGui.QPushButton(self.wizardPage_6)
        self.chemotaxisClearTablePB.setObjectName(_fromUtf8("chemotaxisClearTablePB"))
        self.horizontalLayout_19.addWidget(self.chemotaxisClearTablePB)
        self.verticalLayout_15.addLayout(self.horizontalLayout_19)
        NewSimulationWizard.addPage(self.wizardPage_6)
        self.wizardPage_2 = QtGui.QWizardPage()
        self.wizardPage_2.setObjectName(_fromUtf8("wizardPage_2"))
        self.verticalLayout_7 = QtGui.QVBoxLayout(self.wizardPage_2)
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.label_8 = QtGui.QLabel(self.wizardPage_2)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.verticalLayout_7.addWidget(self.label_8)
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.afTable = QtGui.QTableWidget(self.wizardPage_2)
        self.afTable.setEnabled(True)
        self.afTable.setBaseSize(QtCore.QSize(256, 171))
        self.afTable.setObjectName(_fromUtf8("afTable"))
        self.afTable.setColumnCount(1)
        self.afTable.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.afTable.setHorizontalHeaderItem(0, item)
        self.horizontalLayout_8.addWidget(self.afTable)
        self.clearAFTablePB = QtGui.QPushButton(self.wizardPage_2)
        self.clearAFTablePB.setObjectName(_fromUtf8("clearAFTablePB"))
        self.horizontalLayout_8.addWidget(self.clearAFTablePB)
        spacerItem13 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem13)
        self.verticalLayout_7.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.label_9 = QtGui.QLabel(self.wizardPage_2)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.horizontalLayout_7.addWidget(self.label_9)
        self.afMoleculeLE = QtGui.QLineEdit(self.wizardPage_2)
        self.afMoleculeLE.setText(_fromUtf8(""))
        self.afMoleculeLE.setObjectName(_fromUtf8("afMoleculeLE"))
        self.horizontalLayout_7.addWidget(self.afMoleculeLE)
        self.afMoleculeAddPB = QtGui.QPushButton(self.wizardPage_2)
        self.afMoleculeAddPB.setObjectName(_fromUtf8("afMoleculeAddPB"))
        self.horizontalLayout_7.addWidget(self.afMoleculeAddPB)
        self.verticalLayout_7.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.label_10 = QtGui.QLabel(self.wizardPage_2)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.horizontalLayout_6.addWidget(self.label_10)
        self.bindingFormulaLE = QtGui.QLineEdit(self.wizardPage_2)
        self.bindingFormulaLE.setObjectName(_fromUtf8("bindingFormulaLE"))
        self.horizontalLayout_6.addWidget(self.bindingFormulaLE)
        self.verticalLayout_7.addLayout(self.horizontalLayout_6)
        NewSimulationWizard.addPage(self.wizardPage_2)
        self.wizardPage_4 = QtGui.QWizardPage()
        self.wizardPage_4.setObjectName(_fromUtf8("wizardPage_4"))
        self.verticalLayout_8 = QtGui.QVBoxLayout(self.wizardPage_4)
        self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
        self.label_12 = QtGui.QLabel(self.wizardPage_4)
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.verticalLayout_8.addWidget(self.label_12)
        self.horizontalLayout_10 = QtGui.QHBoxLayout()
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        self.cmcTable = QtGui.QTableWidget(self.wizardPage_4)
        self.cmcTable.setEnabled(True)
        self.cmcTable.setBaseSize(QtCore.QSize(256, 171))
        self.cmcTable.setObjectName(_fromUtf8("cmcTable"))
        self.cmcTable.setColumnCount(1)
        self.cmcTable.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.cmcTable.setHorizontalHeaderItem(0, item)
        self.horizontalLayout_10.addWidget(self.cmcTable)
        self.clearCMCTablePB = QtGui.QPushButton(self.wizardPage_4)
        self.clearCMCTablePB.setObjectName(_fromUtf8("clearCMCTablePB"))
        self.horizontalLayout_10.addWidget(self.clearCMCTablePB)
        spacerItem14 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem14)
        self.verticalLayout_8.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.label_11 = QtGui.QLabel(self.wizardPage_4)
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.horizontalLayout_9.addWidget(self.label_11)
        self.cmcMoleculeLE = QtGui.QLineEdit(self.wizardPage_4)
        self.cmcMoleculeLE.setText(_fromUtf8(""))
        self.cmcMoleculeLE.setObjectName(_fromUtf8("cmcMoleculeLE"))
        self.horizontalLayout_9.addWidget(self.cmcMoleculeLE)
        self.cmcMoleculeAddPB = QtGui.QPushButton(self.wizardPage_4)
        self.cmcMoleculeAddPB.setObjectName(_fromUtf8("cmcMoleculeAddPB"))
        self.horizontalLayout_9.addWidget(self.cmcMoleculeAddPB)
        self.verticalLayout_8.addLayout(self.horizontalLayout_9)
        NewSimulationWizard.addPage(self.wizardPage_4)
        self.wizardPage_5 = QtGui.QWizardPage()
        self.wizardPage_5.setObjectName(_fromUtf8("wizardPage_5"))
        self.verticalLayout_10 = QtGui.QVBoxLayout(self.wizardPage_5)
        self.verticalLayout_10.setObjectName(_fromUtf8("verticalLayout_10"))
        self.label_13 = QtGui.QLabel(self.wizardPage_5)
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.verticalLayout_10.addWidget(self.label_13)
        self.horizontalLayout_13 = QtGui.QHBoxLayout()
        self.horizontalLayout_13.setObjectName(_fromUtf8("horizontalLayout_13"))
        self.groupBox_3 = QtGui.QGroupBox(self.wizardPage_5)
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.verticalLayout_9 = QtGui.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_9.setObjectName(_fromUtf8("verticalLayout_9"))
        self.horizontalLayout_11 = QtGui.QHBoxLayout()
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        self.dictCB = QtGui.QCheckBox(self.groupBox_3)
        self.dictCB.setObjectName(_fromUtf8("dictCB"))
        self.horizontalLayout_11.addWidget(self.dictCB)
        self.listCB = QtGui.QCheckBox(self.groupBox_3)
        self.listCB.setObjectName(_fromUtf8("listCB"))
        self.horizontalLayout_11.addWidget(self.listCB)
        self.verticalLayout_9.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_13.addWidget(self.groupBox_3)
        spacerItem15 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_13.addItem(spacerItem15)
        self.verticalLayout_10.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_14 = QtGui.QHBoxLayout()
        self.horizontalLayout_14.setObjectName(_fromUtf8("horizontalLayout_14"))
        self.plotTable = QtGui.QTableWidget(self.wizardPage_5)
        self.plotTable.setEnabled(True)
        self.plotTable.setBaseSize(QtCore.QSize(256, 171))
        self.plotTable.setObjectName(_fromUtf8("plotTable"))
        self.plotTable.setColumnCount(2)
        self.plotTable.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.plotTable.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.plotTable.setHorizontalHeaderItem(1, item)
        self.horizontalLayout_14.addWidget(self.plotTable)
        self.clearPlotTablePB = QtGui.QPushButton(self.wizardPage_5)
        self.clearPlotTablePB.setObjectName(_fromUtf8("clearPlotTablePB"))
        self.horizontalLayout_14.addWidget(self.clearPlotTablePB)
        spacerItem16 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem16)
        self.verticalLayout_10.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_12 = QtGui.QHBoxLayout()
        self.horizontalLayout_12.setObjectName(_fromUtf8("horizontalLayout_12"))
        self.label_14 = QtGui.QLabel(self.wizardPage_5)
        self.label_14.setObjectName(_fromUtf8("label_14"))
        self.horizontalLayout_12.addWidget(self.label_14)
        self.plotLE = QtGui.QLineEdit(self.wizardPage_5)
        self.plotLE.setObjectName(_fromUtf8("plotLE"))
        self.horizontalLayout_12.addWidget(self.plotLE)
        self.label_15 = QtGui.QLabel(self.wizardPage_5)
        self.label_15.setObjectName(_fromUtf8("label_15"))
        self.horizontalLayout_12.addWidget(self.label_15)
        self.plotTypeCB = QtGui.QComboBox(self.wizardPage_5)
        self.plotTypeCB.setMinimumSize(QtCore.QSize(150, 0))
        self.plotTypeCB.setObjectName(_fromUtf8("plotTypeCB"))
        self.plotTypeCB.addItem(_fromUtf8(""))
        self.plotTypeCB.addItem(_fromUtf8(""))
        self.plotTypeCB.addItem(_fromUtf8(""))
        self.plotTypeCB.addItem(_fromUtf8(""))
        self.horizontalLayout_12.addWidget(self.plotTypeCB)
        self.plotAddPB = QtGui.QPushButton(self.wizardPage_5)
        self.plotAddPB.setObjectName(_fromUtf8("plotAddPB"))
        self.horizontalLayout_12.addWidget(self.plotAddPB)
        self.verticalLayout_10.addLayout(self.horizontalLayout_12)
        NewSimulationWizard.addPage(self.wizardPage_5)
        self.wizardPage_3 = QtGui.QWizardPage()
        self.wizardPage_3.setObjectName(_fromUtf8("wizardPage_3"))
        self.label_7 = QtGui.QLabel(self.wizardPage_3)
        self.label_7.setGeometry(QtCore.QRect(9, 9, 228, 25))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.textBrowser = QtGui.QTextBrowser(self.wizardPage_3)
        self.textBrowser.setGeometry(QtCore.QRect(20, 40, 391, 192))
        self.textBrowser.setFrameShape(QtGui.QFrame.NoFrame)
        self.textBrowser.setFrameShadow(QtGui.QFrame.Plain)
        self.textBrowser.setObjectName(_fromUtf8("textBrowser"))
        NewSimulationWizard.addPage(self.wizardPage_3)

        self.retranslateUi(NewSimulationWizard)
        QtCore.QObject.connect(self.piffRB, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.piffPB.setShown)
        QtCore.QObject.connect(self.piffRB, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.piffLE.setShown)
        QtCore.QMetaObject.connectSlotsByName(NewSimulationWizard)

    def retranslateUi(self, NewSimulationWizard):
        NewSimulationWizard.setWindowTitle(_translate("NewSimulationWizard", "Simulation Wizard", None))
        self.label.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; font-weight:600; color:#0000ff;\">CompuCell3D Simulation Wizard</span></p></body></html>", None))
        self.label_2.setText(_translate("NewSimulationWizard", "Simulation Name", None))
        self.nameLE.setText(_translate("NewSimulationWizard", "NewSimulation", None))
        self.label_3.setText(_translate("NewSimulationWizard", "Simulation Directory", None))
        self.dirLE.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Directory where simulation files will be written to. E.g. if you pick <span style=\" font-weight:600;\">C:\\CC3DProject</span>s then simulation files will be written to directory <span style=\" font-weight:600;\">C:\\CC3DProjects\\&lt;SimulationName&gt;</span></p></body></html>", None))
        self.dirPB.setText(_translate("NewSimulationWizard", "Browse...", None))
        self.groupBox.setTitle(_translate("NewSimulationWizard", "Simulation Type", None))
        self.xmlRB.setToolTip(_translate("NewSimulationWizard", "Only XML Script will be generated", None))
        self.xmlRB.setText(_translate("NewSimulationWizard", "XML only", None))
        self.pythonXMLRB.setToolTip(_translate("NewSimulationWizard", "The following files will be generated:\n"
"1. XMLScript\n"
"2. Python Main script\n"
"3. Python Steppable File\n"
"", None))
        self.pythonXMLRB.setText(_translate("NewSimulationWizard", "Python+XML", None))
        self.pythonOnlyRB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">The following files will be generated:</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">1. Python Main script (including configureSim function which replaces XML )</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">2. Python Steppable File</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt;\"></p></body></html>", None))
        self.pythonOnlyRB.setText(_translate("NewSimulationWizard", "Python only", None))
        self.label_27.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#0000ff;\">General Simulation Properties</span></p></body></html>", None))
        self.groupBox_7.setTitle(_translate("NewSimulationWizard", "Lattice Dimensions", None))
        self.label_28.setText(_translate("NewSimulationWizard", "x", None))
        self.label_29.setText(_translate("NewSimulationWizard", "y", None))
        self.label_30.setText(_translate("NewSimulationWizard", "z", None))
        self.label_31.setText(_translate("NewSimulationWizard", "Average Membrane Fluctuations", None))
        self.membraneFluctuationsLE.setToolTip(_translate("NewSimulationWizard", "Also known as so called temperature parameter in the acceptance probability expresion:  exp(-delta E/(k*T))", None))
        self.membraneFluctuationsLE.setText(_translate("NewSimulationWizard", "10", None))
        self.label_32.setText(_translate("NewSimulationWizard", "Pixel Copy Range (NeighborOrder)", None))
        self.label_33.setText(_translate("NewSimulationWizard", "Number of MC Steps", None))
        self.latticeTypeCB.setItemText(0, _translate("NewSimulationWizard", "Square", None))
        self.latticeTypeCB.setItemText(1, _translate("NewSimulationWizard", "Hexagonal", None))
        self.label_44.setText(_translate("NewSimulationWizard", "LatticeType", None))
        self.groupBox_8.setTitle(_translate("NewSimulationWizard", "Initial Cell Layout", None))
        self.uniformRB.setText(_translate("NewSimulationWizard", "Rectangular Slab", None))
        self.blobRB.setText(_translate("NewSimulationWizard", "Blob", None))
        self.piffRB.setText(_translate("NewSimulationWizard", "Custom Layout (PIFF file)", None))
        self.piffPB.setText(_translate("NewSimulationWizard", "PIFF file...", None))
        self.label_5.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#0000ff;\">Cell Type Specification</span></p></body></html>", None))
        item = self.cellTypeTable.horizontalHeaderItem(0)
        item.setText(_translate("NewSimulationWizard", "Cell Type", None))
        item = self.cellTypeTable.horizontalHeaderItem(1)
        item.setText(_translate("NewSimulationWizard", "Freeze", None))
        self.clearCellTypeTablePB.setText(_translate("NewSimulationWizard", "Clear Table", None))
        self.label_4.setText(_translate("NewSimulationWizard", "Cell Type", None))
        self.freezeCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Indicates whether cells of this type should remain frozen during simulation</span></p></body></html>", None))
        self.freezeCHB.setText(_translate("NewSimulationWizard", "Freeze", None))
        self.cellTypeAddPB.setText(_translate("NewSimulationWizard", "Add", None))
        self.label_17.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#0000ff;\">Chemical Fields (diffusants)</span></p></body></html>", None))
        item = self.fieldTable.horizontalHeaderItem(0)
        item.setText(_translate("NewSimulationWizard", "Field Name", None))
        item = self.fieldTable.horizontalHeaderItem(1)
        item.setText(_translate("NewSimulationWizard", "Solver", None))
        self.clearFieldTablePB.setText(_translate("NewSimulationWizard", "Clear Table", None))
        self.pdeSolverCallerCHB.setToolTip(_translate("NewSimulationWizard", "Inserts PDESolver plugin into generated code. ", None))
        self.pdeSolverCallerCHB.setText(_translate("NewSimulationWizard", "Enable multiple calls of PDE solvers", None))
        self.label_18.setText(_translate("NewSimulationWizard", "Field Name", None))
        self.label_19.setText(_translate("NewSimulationWizard", "Solver", None))
        self.solverCB.setItemText(0, _translate("NewSimulationWizard", "DiffusionSolverFE", None))
        self.solverCB.setItemText(1, _translate("NewSimulationWizard", "FlexibleDiffusionSolverFE", None))
        self.solverCB.setItemText(2, _translate("NewSimulationWizard", "FastDiffusionSolver2DFE", None))
        self.solverCB.setItemText(3, _translate("NewSimulationWizard", "KernelDiffusionSolver", None))
        self.solverCB.setItemText(4, _translate("NewSimulationWizard", "SteadyStateDiffusionSolver", None))
        self.fieldAddPB.setText(_translate("NewSimulationWizard", "Add", None))
        self.label_6.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#0000ff;\">Cell Properties and Behaviors</span></p></body></html>", None))
        self.groupBox_9.setTitle(_translate("NewSimulationWizard", "Cellular Behaviors", None))
        self.label_34.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600; text-decoration: underline;\">Adhesion</span></p></body></html>", None))
        self.contactCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Most commonly used energy term for contact (adhesive) cell-cell/cell-Medium interactions</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Note (when using compartmental cells): it calculates energy between members (compartments) belonging to different clusters (compartmental cells).</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt;\"></p></body></html>", None))
        self.contactCHB.setText(_translate("NewSimulationWizard", "Contact", None))
        self.internalContactCB.setToolTip(_translate("NewSimulationWizard", "Adhesion energy term - calculated betwee members (compartments) of the same cluster (compartmental cell)\n"
"You may use it together with Contact energy term", None))
        self.internalContactCB.setText(_translate("NewSimulationWizard", "ContactInternal", None))
        self.adhesionFlexCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Calculates adhesion energy based on concentration of adhesion moolecules on the cell membrane. Works fine for complartmental and non compartmental cells. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Users can specifymultiple adhesion molecules and customize interactions between them. Adhesion molecules concentrations can be modified/accessed using Python scripting</span></p></body></html>", None))
        self.adhesionFlexCHB.setText(_translate("NewSimulationWizard", "AdhesionFlex", None))
        self.contactLocalProductCHB.setToolTip(_translate("NewSimulationWizard", "Older version of AdhesionFlex. Please consider switching to AdhesionFlex plugin", None))
        self.contactLocalProductCHB.setText(_translate("NewSimulationWizard", "ContactLocalProduct", None))
        self.compartmentCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Combined Contact and ContactInternal plugin.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Allows users to specify adhesions between members of same and different clusters at the same time.</span></p></body></html>", None))
        self.compartmentCHB.setText(_translate("NewSimulationWizard", "Compartments", None))
        self.fppCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Emulates focal junctions by dynamically linking (via elastic constraint) center of masses of neighboring cells. Elastic constraint is based on the distance between linked cells. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Links\' parameters can be accessed and modified through Python.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Links can be brokeneither due to exceeding max distance between cells or manually.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Links are established when two cells come in contact and max number of links for the two cells is smaller than the number of links they have already formed.</span></p></body></html>", None))
        self.fppCHB.setText(_translate("NewSimulationWizard", "FocalPointPlasticity", None))
        self.elasticityCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Applies elastic constraint between cell\'s center of masses of participating cells. Elastic constraint is based on the distance between linked cells. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Elastic links are initially formed between those cells which touch each other when first pixel copy is about to happen. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Links can be later modified, added/removed using Python scripting. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">The list of links is static i.e. once two cells are linked they will remain linked until one of them disappears or the linkis broken manually by Python script.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt;\"></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt;\"></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt;\"></p></body></html>", None))
        self.elasticityCHB.setText(_translate("NewSimulationWizard", "Elasticity", None))
        self.contactMultiCadCHB.setToolTip(_translate("NewSimulationWizard", "Deprecetad plauing. Please use AdhesionFlex", None))
        self.contactMultiCadCHB.setText(_translate("NewSimulationWizard", "ContactMultiCad (deprecated)", None))
        self.label_35.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600; text-decoration: underline;\">Chemotaxis</span></p></body></html>", None))
        self.chemotaxisCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Implements energy term which emulates Chemotaxis. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Note, you need to define chemical fields for this plugin to work.</span></p></body></html>", None))
        self.chemotaxisCHB.setText(_translate("NewSimulationWizard", "Chemotaxis", None))
        self.label_36.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600; text-decoration: underline;\">Secretion</span></p></body></html>", None))
        self.secretionCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">A module which implements secretion. You need to make sure you have defined chemical fields to which you are trying to secrete. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Secretion can be deinied for particular cell types using XML/Python configureSim  or on a cell-by-cell basis using Python </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Notice it implicitely calls PixelTracker and BoundaryPixelTracker plugins</span></p></body></html>", None))
        self.secretionCHB.setText(_translate("NewSimulationWizard", "Secretion", None))
        self.label_37.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600; text-decoration: underline;\">Growth</span></p></body></html>", None))
        self.growthCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Example of a Python steppable which implements cell growth.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\"> Users can modify the code to refine cell growth behavior. </span></p></body></html>", None))
        self.growthCHB.setText(_translate("NewSimulationWizard", "Growth (Python)", None))
        self.label_38.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600; text-decoration: underline;\">Mitosis</span></p></body></html>", None))
        self.mitosisCHB.setToolTip(_translate("NewSimulationWizard", "Python Steppable implementing cell division. ", None))
        self.mitosisCHB.setText(_translate("NewSimulationWizard", "Mitosis (Python)", None))
        self.label_39.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600; text-decoration: underline;\">Death</span></p></body></html>", None))
        self.deathCHB.setToolTip(_translate("NewSimulationWizard", "Python Steppable implementing cell death", None))
        self.deathCHB.setText(_translate("NewSimulationWizard", "Death (Python)", None))
        self.groupBox_2.setTitle(_translate("NewSimulationWizard", "Constraints and Forces", None))
        self.label_40.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600; text-decoration: underline;\">Volume</span></p></body></html>", None))
        self.volumeFlexCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Volume constraint energy term. E=Lambda*(v-V</span><span style=\" font-size:8pt; vertical-align:sub;\">T</span><span style=\" font-size:8pt;\">)</span><span style=\" font-size:8pt; vertical-align:super;\">2 </span><span style=\" font-size:8pt;\">. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Lambda and V</span><span style=\" font-size:8pt; vertical-align:sub;\">T </span><span style=\" font-size:8pt;\">are</span><span style=\" font-size:8pt; vertical-align:super;\"> </span><span style=\" font-size:8pt;\">specified for each cell type separately</span></p></body></html>", None))
        self.volumeFlexCHB.setText(_translate("NewSimulationWizard", "VolumeFlex", None))
        self.volumeLocalFlexCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Volume constraint energy term. E=Lambda*(v-V</span><span style=\" font-size:8pt; vertical-align:sub;\">T</span><span style=\" font-size:8pt;\">)</span><span style=\" font-size:8pt; vertical-align:super;\">2 </span><span style=\" font-size:8pt;\">.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\"> Lambda and V</span><span style=\" font-size:8pt; vertical-align:sub;\">T </span><span style=\" font-size:8pt;\">are</span><span style=\" font-size:8pt; vertical-align:super;\"> </span><span style=\" font-size:8pt;\">specified for each cell separately. Lambda and target volume are specified in Python.</span></p></body></html>", None))
        self.volumeLocalFlexCHB.setText(_translate("NewSimulationWizard", "VolumeLocalFlex", None))
        self.label_41.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600; text-decoration: underline;\">Surface</span></p></body></html>", None))
        self.surfaceFlexCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Surface constraint energy term. E=Lambda*(s-S</span><span style=\" font-size:8pt; vertical-align:sub;\">T</span><span style=\" font-size:8pt;\">)</span><span style=\" font-size:8pt; vertical-align:super;\">2 </span><span style=\" font-size:8pt;\">. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Lambda and S</span><span style=\" font-size:8pt; vertical-align:sub;\">T </span><span style=\" font-size:8pt;\">are</span><span style=\" font-size:8pt; vertical-align:super;\"> </span><span style=\" font-size:8pt;\">specified for each cell type separately</span></p></body></html>", None))
        self.surfaceFlexCHB.setText(_translate("NewSimulationWizard", "SurfaceFlex", None))
        self.surfaceLocalFlexCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Surface constraint energy term. E=Lambda*(s-S</span><span style=\" font-size:8pt; vertical-align:sub;\">T</span><span style=\" font-size:8pt;\">)</span><span style=\" font-size:8pt; vertical-align:super;\">2 </span><span style=\" font-size:8pt;\">. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Lambda and S</span><span style=\" font-size:8pt; vertical-align:sub;\">T </span><span style=\" font-size:8pt;\">are</span><span style=\" font-size:8pt; vertical-align:super;\"> </span><span style=\" font-size:8pt;\">specified for each cell separately. Lambda and target surface are specified in Python.</span></p></body></html>", None))
        self.surfaceLocalFlexCHB.setText(_translate("NewSimulationWizard", "SurfaceLocalFlex", None))
        self.label_42.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600; text-decoration: underline;\">Ext. Force</span></p></body></html>", None))
        self.extPotCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Energy term emulating force applied to cells. Force components are specified for cell types.</span></p></body></html>", None))
        self.extPotCHB.setText(_translate("NewSimulationWizard", "ExternalPotential", None))
        self.extPotLocalFlexCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Energy term emulating force applied to cells. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Force components are specified (in Python) for each cell individuelly</span></p></body></html>", None))
        self.extPotLocalFlexCHB.setText(_translate("NewSimulationWizard", "ExternalPotentialLocalFlex", None))
        self.label_43.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600; text-decoration: underline;\">Connectivity</span></p></body></html>", None))
        self.connectGlobalCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Energy term which adds high penalty to change of energy when cells are about to fragment. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">It ensures that cells remain connected . </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Works in 2D and 3D and on square and hex lattice. Energy penalty specifications are on a per-cell type basis.</span></p></body></html>", None))
        self.connectGlobalCHB.setText(_translate("NewSimulationWizard", "Global (2D/3D)", None))
        self.connectGlobalByIdCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Energy term which adds high penalty to change of energy when cells are about to fragment. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">It ensures that cells remain connected. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Works in 2D and 3D and on square and hex lattice. Energy penalty specifications are on a per-cell basis and thus require Python scripting.</span></p></body></html>", None))
        self.connectGlobalByIdCHB.setText(_translate("NewSimulationWizard", "Global (by cell id)", None))
        self.connect2DCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Energy term which adds high penalty to change of energy when cells are about to fragment. It ensures that cells remain connected. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Works only in  2D on square. Energy penalty specification is global for all cells. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">There is a more flexible version of this plugin which allows specification of penalties on a per-cell basis it is called ConnectivityLocalFlex and reuqires Python scripting</span></p></body></html>", None))
        self.connect2DCHB.setText(_translate("NewSimulationWizard", "Fast (2D square lattice)", None))
        self.label_45.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600; text-decoration: underline;\">Elongation</span></p></body></html>", None))
        self.lengthConstraintCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Energy term which favors pixel copies that elongate cells according to specified paramters.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">This constraint often requires connectivity constraint for large cell elongations.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Parameter specifications are on per-cell type basis.</span></p></body></html>", None))
        self.lengthConstraintCHB.setText(_translate("NewSimulationWizard", "LengthConstraint", None))
        self.lengthConstraintLocalFlexCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Energy term which favors pixel copies that elongate cells according to specified paramters.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">This constraint often requires connectivity constraint for large cell elongations.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Parameter specifications are on per-cell basis and require Python scripting. </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">This energy term works only in 2D</span></p></body></html>", None))
        self.lengthConstraintLocalFlexCHB.setText(_translate("NewSimulationWizard", "LengthConstraintLocalFlex (2D)", None))
        self.groupBox_4.setTitle(_translate("NewSimulationWizard", "Cellular Property Trackers", None))
        self.comCHB.setToolTip(_translate("NewSimulationWizard", "Module tracking center of mass of each cell in the smiulation", None))
        self.comCHB.setText(_translate("NewSimulationWizard", "Center Of Mass", None))
        self.neighborCHB.setToolTip(_translate("NewSimulationWizard", "Module tracking neighbors of each cell", None))
        self.neighborCHB.setText(_translate("NewSimulationWizard", "Cell Neighbors", None))
        self.momentOfInertiaCHB.setToolTip(_translate("NewSimulationWizard", "Module tracking tensor of inertia of each cell", None))
        self.momentOfInertiaCHB.setText(_translate("NewSimulationWizard", "Moment Of Inertia", None))
        self.pixelTrackerCHB.setToolTip(_translate("NewSimulationWizard", "Module tracking and storing pixels of each cell", None))
        self.pixelTrackerCHB.setText(_translate("NewSimulationWizard", "Cell Pixel Tracker", None))
        self.boundaryPixelTrackerCHB.setToolTip(_translate("NewSimulationWizard", "Module tracking and storing those pixels of a cell which are at the cell boundary", None))
        self.boundaryPixelTrackerCHB.setText(_translate("NewSimulationWizard", "Cell Boundary Pixel Tracker", None))
        self.groupBox_10.setTitle(_translate("NewSimulationWizard", "Aux. Modules", None))
        self.boxWatcherCHB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Module computing minimal box enclosing all cells in the simulation.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Can be used to speed up some simulations</span></p></body></html>", None))
        self.boxWatcherCHB.setText(_translate("NewSimulationWizard", "BoxWatcher", None))
        self.pifDumperCHB.setToolTip(_translate("NewSimulationWizard", "Module which outputs periodically cell lattice in thee PIFF format", None))
        self.pifDumperCHB.setText(_translate("NewSimulationWizard", "PIFDumper", None))
        self.label_24.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#0000ff;\">Secretion Plugin</span></p></body></html>", None))
        item = self.secretionTable.horizontalHeaderItem(0)
        item.setText(_translate("NewSimulationWizard", "Field", None))
        item = self.secretionTable.horizontalHeaderItem(1)
        item.setText(_translate("NewSimulationWizard", "CellType", None))
        item = self.secretionTable.horizontalHeaderItem(2)
        item.setText(_translate("NewSimulationWizard", "Rate", None))
        item = self.secretionTable.horizontalHeaderItem(3)
        item.setText(_translate("NewSimulationWizard", "On Contact With", None))
        item = self.secretionTable.horizontalHeaderItem(4)
        item.setText(_translate("NewSimulationWizard", "Type", None))
        self.groupBox_6.setTitle(_translate("NewSimulationWizard", "Secretion Type", None))
        self.secrUniformRB.setText(_translate("NewSimulationWizard", "uniform", None))
        self.secrOnContactRB.setText(_translate("NewSimulationWizard", "on contact", None))
        self.secrConstConcRB.setText(_translate("NewSimulationWizard", "constant concentration", None))
        self.label_25.setText(_translate("NewSimulationWizard", "Field", None))
        self.label_26.setText(_translate("NewSimulationWizard", "Cell type", None))
        self.secrRateLB.setText(_translate("NewSimulationWizard", "Secretion Rate", None))
        self.secrAddOnContactPB.setText(_translate("NewSimulationWizard", "Add On Contact", None))
        self.secrAddRowPB.setText(_translate("NewSimulationWizard", "Add Entry", None))
        self.secrRemoveRowsPB.setText(_translate("NewSimulationWizard", "Remove Rows", None))
        self.secrClearTablePB.setText(_translate("NewSimulationWizard", "Clear Table", None))
        self.label_16.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#0000ff;\">Chemotaxis Plugin</span></p></body></html>", None))
        item = self.chamotaxisTable.horizontalHeaderItem(0)
        item.setText(_translate("NewSimulationWizard", "Field", None))
        item = self.chamotaxisTable.horizontalHeaderItem(1)
        item.setText(_translate("NewSimulationWizard", "CellType", None))
        item = self.chamotaxisTable.horizontalHeaderItem(2)
        item.setText(_translate("NewSimulationWizard", "Lambda", None))
        item = self.chamotaxisTable.horizontalHeaderItem(3)
        item.setText(_translate("NewSimulationWizard", "ChemotaxTowards", None))
        item = self.chamotaxisTable.horizontalHeaderItem(4)
        item.setText(_translate("NewSimulationWizard", "Sat. Coef.", None))
        item = self.chamotaxisTable.horizontalHeaderItem(5)
        item.setText(_translate("NewSimulationWizard", "Type", None))
        self.groupBox_5.setTitle(_translate("NewSimulationWizard", "Chemotaxis Type", None))
        self.chemRegRB.setText(_translate("NewSimulationWizard", "regular", None))
        self.chemSatRB.setText(_translate("NewSimulationWizard", "saturation", None))
        self.chemSatLinRB.setText(_translate("NewSimulationWizard", "saturation linear", None))
        self.label_20.setText(_translate("NewSimulationWizard", "Field", None))
        self.label_21.setText(_translate("NewSimulationWizard", "Cell type", None))
        self.label_22.setText(_translate("NewSimulationWizard", "Lambda", None))
        self.satCoefLB.setText(_translate("NewSimulationWizard", "Saturation Coef.", None))
        self.chemotaxTowardsPB.setText(_translate("NewSimulationWizard", "Chemotax Towards", None))
        self.label_23.setText(_translate("NewSimulationWizard", "Cell Type", None))
        self.chemotaxisAddRowPB.setText(_translate("NewSimulationWizard", "Add Entry", None))
        self.chemotaxisRemoveRowsPB.setText(_translate("NewSimulationWizard", "Remove Rows", None))
        self.chemotaxisClearTablePB.setText(_translate("NewSimulationWizard", "Clear Table", None))
        self.label_8.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#0000ff;\">AdhesionFlex Plugin</span></p></body></html>", None))
        item = self.afTable.horizontalHeaderItem(0)
        item.setText(_translate("NewSimulationWizard", "Adhesion Molecule", None))
        self.clearAFTablePB.setText(_translate("NewSimulationWizard", "Clear Table", None))
        self.label_9.setText(_translate("NewSimulationWizard", "Molecule", None))
        self.afMoleculeLE.setToolTip(_translate("NewSimulationWizard", "Specify names of the adhesion molecules you want to use int he simulation", None))
        self.afMoleculeAddPB.setText(_translate("NewSimulationWizard", "Add", None))
        self.label_10.setText(_translate("NewSimulationWizard", "Binding Formula", None))
        self.bindingFormulaLE.setToolTip(_translate("NewSimulationWizard", "This is binary function that takes atwo arguments -  Molecule1 and Molecule2. The allowed functions are those given by muParser - see http://muparser.sourceforge.net/", None))
        self.bindingFormulaLE.setText(_translate("NewSimulationWizard", "min(Molecule1,Molecule2)", None))
        self.label_12.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#0000ff;\">ContactMultiCad Plugin</span></p></body></html>", None))
        item = self.cmcTable.horizontalHeaderItem(0)
        item.setText(_translate("NewSimulationWizard", "Cadherin", None))
        self.clearCMCTablePB.setText(_translate("NewSimulationWizard", "Clear Table", None))
        self.label_11.setText(_translate("NewSimulationWizard", "Cadherin", None))
        self.cmcMoleculeLE.setToolTip(_translate("NewSimulationWizard", "Specify names of the adhesion molecules you want to use int he simulation", None))
        self.cmcMoleculeAddPB.setText(_translate("NewSimulationWizard", "Add", None))
        self.label_13.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#0000ff;\">Python script options</span></p></body></html>", None))
        self.groupBox_3.setTitle(_translate("NewSimulationWizard", "Cell Python Attributes", None))
        self.dictCB.setToolTip(_translate("NewSimulationWizard", "Generates code which attaches Python dictionary to each cell. The dictionary can be accessed/ modified in Python.", None))
        self.dictCB.setText(_translate("NewSimulationWizard", "Dictionary", None))
        self.listCB.setToolTip(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Generates code which attaches Python list to each cell. The list can be accessed/ modified in Python.</span></p></body></html>", None))
        self.listCB.setText(_translate("NewSimulationWizard", "List", None))
        item = self.plotTable.horizontalHeaderItem(0)
        item.setText(_translate("NewSimulationWizard", "Plot Name", None))
        item = self.plotTable.horizontalHeaderItem(1)
        item.setText(_translate("NewSimulationWizard", "Type", None))
        self.clearPlotTablePB.setText(_translate("NewSimulationWizard", "Clear Table", None))
        self.label_14.setText(_translate("NewSimulationWizard", "Plot Name", None))
        self.label_15.setText(_translate("NewSimulationWizard", "Type", None))
        self.plotTypeCB.setItemText(0, _translate("NewSimulationWizard", "ScalarField", None))
        self.plotTypeCB.setItemText(1, _translate("NewSimulationWizard", "CellLevelScalarField", None))
        self.plotTypeCB.setItemText(2, _translate("NewSimulationWizard", "VectorField", None))
        self.plotTypeCB.setItemText(3, _translate("NewSimulationWizard", "CellLevelVectorField", None))
        self.plotAddPB.setText(_translate("NewSimulationWizard", "Add", None))
        self.label_7.setText(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#0000ff;\">Configuration Complete!</span></p></body></html>", None))
        self.textBrowser.setHtml(_translate("NewSimulationWizard", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">CC3D project will be generated now</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600; color:#0000ff;\">NOTE:</span><span style=\" font-size:12pt;\"> The parameters in the XML and Python scripts will have to be changed to be realistic. Please see CC3D manual on how to choose simulation parameters</span></p></body></html>", None))

