# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'SteppableGeneratorDialog.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_SteppableGenerator(object):

    def setupUi(self, SteppableGenerator):

        SteppableGenerator.setObjectName("SteppableGenerator")

        SteppableGenerator.resize(421, 323)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(SteppableGenerator)

        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.verticalLayout = QtWidgets.QVBoxLayout()

        self.verticalLayout.setObjectName("verticalLayout")

        self.label_2 = QtWidgets.QLabel(SteppableGenerator)

        self.label_2.setObjectName("label_2")

        self.verticalLayout.addWidget(self.label_2)

        self.mainScriptLB = QtWidgets.QLabel(SteppableGenerator)

        self.mainScriptLB.setObjectName("mainScriptLB")

        self.verticalLayout.addWidget(self.mainScriptLB)

        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.gridLayout = QtWidgets.QGridLayout()

        self.gridLayout.setObjectName("gridLayout")

        self.label = QtWidgets.QLabel(SteppableGenerator)

        self.label.setObjectName("label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.steppebleNameLE = QtWidgets.QLineEdit(SteppableGenerator)

        self.steppebleNameLE.setObjectName("steppebleNameLE")

        self.gridLayout.addWidget(self.steppebleNameLE, 0, 1, 1, 2)

        self.label_3 = QtWidgets.QLabel(SteppableGenerator)

        self.label_3.setObjectName("label_3")

        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout.addItem(spacerItem, 1, 1, 1, 1)

        self.freqSB = QtWidgets.QSpinBox(SteppableGenerator)

        self.freqSB.setMinimum(1)

        self.freqSB.setMaximum(10000)

        self.freqSB.setSingleStep(1)

        self.freqSB.setProperty("value", 1)

        self.freqSB.setObjectName("freqSB")

        self.gridLayout.addWidget(self.freqSB, 1, 2, 1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout)

        self.groupBox = QtWidgets.QGroupBox(SteppableGenerator)

        self.groupBox.setObjectName("groupBox")

        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox)

        self.horizontalLayout.setObjectName("horizontalLayout")

        self.genericLB = QtWidgets.QRadioButton(self.groupBox)

        self.genericLB.setChecked(True)

        self.genericLB.setObjectName("genericLB")

        self.horizontalLayout.addWidget(self.genericLB)

        self.mitosisRB = QtWidgets.QRadioButton(self.groupBox)

        self.mitosisRB.setObjectName("mitosisRB")

        self.horizontalLayout.addWidget(self.mitosisRB)

        self.clusterMitosisRB = QtWidgets.QRadioButton(self.groupBox)

        self.clusterMitosisRB.setObjectName("clusterMitosisRB")

        self.horizontalLayout.addWidget(self.clusterMitosisRB)

        self.runBeforeMCSRB = QtWidgets.QRadioButton(self.groupBox)

        self.runBeforeMCSRB.setObjectName("runBeforeMCSRB")

        self.horizontalLayout.addWidget(self.runBeforeMCSRB)

        self.verticalLayout_2.addWidget(self.groupBox)

        self.groupBox_2 = QtWidgets.QGroupBox(SteppableGenerator)

        self.groupBox_2.setObjectName("groupBox_2")

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_2)

        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.vectorCB = QtWidgets.QCheckBox(self.groupBox_2)

        self.vectorCB.setObjectName("vectorCB")

        self.horizontalLayout_2.addWidget(self.vectorCB)

        self.scalarCB = QtWidgets.QCheckBox(self.groupBox_2)

        self.scalarCB.setObjectName("scalarCB")

        self.horizontalLayout_2.addWidget(self.scalarCB)

        self.scalarCellLevelCB = QtWidgets.QCheckBox(self.groupBox_2)

        self.scalarCellLevelCB.setObjectName("scalarCellLevelCB")

        self.horizontalLayout_2.addWidget(self.scalarCellLevelCB)

        self.vectorCellLevelCB = QtWidgets.QCheckBox(self.groupBox_2)

        self.vectorCellLevelCB.setObjectName("vectorCellLevelCB")

        self.horizontalLayout_2.addWidget(self.vectorCellLevelCB)

        self.verticalLayout_2.addWidget(self.groupBox_2)

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_3.setObjectName("horizontalLayout_3")

        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(spacerItem1)

        self.okPB = QtWidgets.QPushButton(SteppableGenerator)

        self.okPB.setObjectName("okPB")

        self.horizontalLayout_3.addWidget(self.okPB)

        self.cancelPB = QtWidgets.QPushButton(SteppableGenerator)

        self.cancelPB.setObjectName("cancelPB")

        self.horizontalLayout_3.addWidget(self.cancelPB)

        self.verticalLayout_2.addLayout(self.horizontalLayout_3)



        self.retranslateUi(SteppableGenerator)

        self.cancelPB.clicked.connect(SteppableGenerator.reject)

        QtCore.QMetaObject.connectSlotsByName(SteppableGenerator)



    def retranslateUi(self, SteppableGenerator):

        _translate = QtCore.QCoreApplication.translate

        SteppableGenerator.setWindowTitle(_translate("SteppableGenerator", "Generate Steppable"))

        self.label_2.setText(_translate("SteppableGenerator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"

"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"

"p, li { white-space: pre-wrap; }\n"

"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"

"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600;\">Steppeble Will be registered in :</span></p></body></html>"))

        self.mainScriptLB.setText(_translate("SteppableGenerator", "TextLabel"))

        self.label.setText(_translate("SteppableGenerator", "SteppableName"))

        self.label_3.setText(_translate("SteppableGenerator", "Call Frequency"))

        self.groupBox.setTitle(_translate("SteppableGenerator", "Steppable Type"))

        self.genericLB.setText(_translate("SteppableGenerator", "Generic"))

        self.mitosisRB.setText(_translate("SteppableGenerator", "Mitosis"))

        self.clusterMitosisRB.setText(_translate("SteppableGenerator", "Cluster Mitosis"))

        self.runBeforeMCSRB.setText(_translate("SteppableGenerator", "Run Before MCS (secretion)"))

        self.groupBox_2.setToolTip(_translate("SteppableGenerator", "You can add extra visualization fields here.\n"

"The fields will be managed from Python"))

        self.groupBox_2.setTitle(_translate("SteppableGenerator", "Extra Visualization Fields"))

        self.vectorCB.setText(_translate("SteppableGenerator", "Vector"))

        self.scalarCB.setText(_translate("SteppableGenerator", "Scalar"))

        self.scalarCellLevelCB.setText(_translate("SteppableGenerator", "Scalar Cell Level"))

        self.vectorCellLevelCB.setText(_translate("SteppableGenerator", "Vector Cell Level"))

        self.okPB.setText(_translate("SteppableGenerator", "OK"))

        self.cancelPB.setText(_translate("SteppableGenerator", "Cancel"))



