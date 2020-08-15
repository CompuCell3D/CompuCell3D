# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'potts.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_PottsDlg(object):

    def setupUi(self, PottsDlg):

        PottsDlg.setObjectName("PottsDlg")

        PottsDlg.resize(526, 550)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(PottsDlg)

        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.groupBox_7 = QtWidgets.QGroupBox(PottsDlg)

        self.groupBox_7.setObjectName("groupBox_7")

        self.horizontalLayout_25 = QtWidgets.QHBoxLayout(self.groupBox_7)

        self.horizontalLayout_25.setObjectName("horizontalLayout_25")

        self.horizontalLayout_24 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_24.setObjectName("horizontalLayout_24")

        self.label_28 = QtWidgets.QLabel(self.groupBox_7)

        self.label_28.setObjectName("label_28")

        self.horizontalLayout_24.addWidget(self.label_28)

        self.xDimSB = QtWidgets.QSpinBox(self.groupBox_7)

        self.xDimSB.setMinimum(1)

        self.xDimSB.setMaximum(10000000)

        self.xDimSB.setProperty("value", 100)

        self.xDimSB.setObjectName("xDimSB")

        self.horizontalLayout_24.addWidget(self.xDimSB)

        self.label_29 = QtWidgets.QLabel(self.groupBox_7)

        self.label_29.setObjectName("label_29")

        self.horizontalLayout_24.addWidget(self.label_29)

        self.yDimSB = QtWidgets.QSpinBox(self.groupBox_7)

        self.yDimSB.setMinimum(1)

        self.yDimSB.setMaximum(10000000)

        self.yDimSB.setProperty("value", 100)

        self.yDimSB.setObjectName("yDimSB")

        self.horizontalLayout_24.addWidget(self.yDimSB)

        self.label_30 = QtWidgets.QLabel(self.groupBox_7)

        self.label_30.setObjectName("label_30")

        self.horizontalLayout_24.addWidget(self.label_30)

        self.zDimSB = QtWidgets.QSpinBox(self.groupBox_7)

        self.zDimSB.setMinimum(1)

        self.zDimSB.setMaximum(10000000)

        self.zDimSB.setObjectName("zDimSB")

        self.horizontalLayout_24.addWidget(self.zDimSB)

        self.horizontalLayout_25.addLayout(self.horizontalLayout_24)

        self.verticalLayout_2.addWidget(self.groupBox_7)

        self.groupBox_3 = QtWidgets.QGroupBox(PottsDlg)

        self.groupBox_3.setObjectName("groupBox_3")

        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupBox_3)

        self.verticalLayout_9.setObjectName("verticalLayout_9")

        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_11.setObjectName("horizontalLayout_11")

        self.label_13 = QtWidgets.QLabel(self.groupBox_3)

        self.label_13.setObjectName("label_13")

        self.horizontalLayout_11.addWidget(self.label_13)

        self.xbcCB = QtWidgets.QComboBox(self.groupBox_3)

        self.xbcCB.setObjectName("xbcCB")

        self.xbcCB.addItem("")

        self.xbcCB.addItem("")

        self.horizontalLayout_11.addWidget(self.xbcCB)

        self.label_14 = QtWidgets.QLabel(self.groupBox_3)

        self.label_14.setObjectName("label_14")

        self.horizontalLayout_11.addWidget(self.label_14)

        self.ybcCB = QtWidgets.QComboBox(self.groupBox_3)

        self.ybcCB.setObjectName("ybcCB")

        self.ybcCB.addItem("")

        self.ybcCB.addItem("")

        self.horizontalLayout_11.addWidget(self.ybcCB)

        self.label_15 = QtWidgets.QLabel(self.groupBox_3)

        self.label_15.setObjectName("label_15")

        self.horizontalLayout_11.addWidget(self.label_15)

        self.zbcCB = QtWidgets.QComboBox(self.groupBox_3)

        self.zbcCB.setObjectName("zbcCB")

        self.zbcCB.addItem("")

        self.zbcCB.addItem("")

        self.horizontalLayout_11.addWidget(self.zbcCB)

        self.verticalLayout_9.addLayout(self.horizontalLayout_11)

        self.verticalLayout_2.addWidget(self.groupBox_3)

        self.verticalLayout = QtWidgets.QVBoxLayout()

        self.verticalLayout.setObjectName("verticalLayout")

        self.gridLayout = QtWidgets.QGridLayout()

        self.gridLayout.setObjectName("gridLayout")

        self.label_44 = QtWidgets.QLabel(PottsDlg)

        self.label_44.setObjectName("label_44")

        self.gridLayout.addWidget(self.label_44, 0, 0, 1, 1)

        self.latticeTypeCB = QtWidgets.QComboBox(PottsDlg)

        self.latticeTypeCB.setObjectName("latticeTypeCB")

        self.latticeTypeCB.addItem("")

        self.latticeTypeCB.addItem("")

        self.gridLayout.addWidget(self.latticeTypeCB, 0, 1, 1, 1)

        self.label_31 = QtWidgets.QLabel(PottsDlg)

        self.label_31.setObjectName("label_31")

        self.gridLayout.addWidget(self.label_31, 1, 0, 1, 1)

        self.membraneFluctuationsLE = QtWidgets.QLineEdit(PottsDlg)

        self.membraneFluctuationsLE.setObjectName("membraneFluctuationsLE")

        self.gridLayout.addWidget(self.membraneFluctuationsLE, 1, 1, 1, 1)

        self.label_32 = QtWidgets.QLabel(PottsDlg)

        self.label_32.setObjectName("label_32")

        self.gridLayout.addWidget(self.label_32, 2, 0, 1, 1)

        self.neighborOrderSB = QtWidgets.QSpinBox(PottsDlg)

        self.neighborOrderSB.setMinimum(1)

        self.neighborOrderSB.setMaximum(10)

        self.neighborOrderSB.setProperty("value", 2)

        self.neighborOrderSB.setObjectName("neighborOrderSB")

        self.gridLayout.addWidget(self.neighborOrderSB, 2, 1, 1, 1)

        self.label_33 = QtWidgets.QLabel(PottsDlg)

        self.label_33.setObjectName("label_33")

        self.gridLayout.addWidget(self.label_33, 3, 0, 1, 1)

        self.mcsSB = QtWidgets.QSpinBox(PottsDlg)

        self.mcsSB.setMaximum(1000000000)

        self.mcsSB.setProperty("value", 1000)

        self.mcsSB.setObjectName("mcsSB")

        self.gridLayout.addWidget(self.mcsSB, 3, 1, 1, 1)

        self.label_34 = QtWidgets.QLabel(PottsDlg)

        self.label_34.setObjectName("label_34")

        self.gridLayout.addWidget(self.label_34, 4, 0, 1, 1)

        self.anneal_mcsSB = QtWidgets.QSpinBox(PottsDlg)

        self.anneal_mcsSB.setMaximum(1000000000)

        self.anneal_mcsSB.setProperty("value", 0)

        self.anneal_mcsSB.setObjectName("anneal_mcsSB")

        self.gridLayout.addWidget(self.anneal_mcsSB, 4, 1, 1, 1)

        self.verticalLayout.addLayout(self.gridLayout)

        self.auto_gen_rand_seed_CB = QtWidgets.QCheckBox(PottsDlg)

        self.auto_gen_rand_seed_CB.setObjectName("auto_gen_rand_seed_CB")

        self.verticalLayout.addWidget(self.auto_gen_rand_seed_CB)

        self.verticalLayout_2.addLayout(self.verticalLayout)

        spacerItem = QtWidgets.QSpacerItem(20, 181, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(spacerItem)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(spacerItem1)

        self.horizontalLayout = QtWidgets.QHBoxLayout()

        self.horizontalLayout.setObjectName("horizontalLayout")

        self.okPB = QtWidgets.QPushButton(PottsDlg)

        self.okPB.setObjectName("okPB")

        self.horizontalLayout.addWidget(self.okPB)

        self.cancelPB = QtWidgets.QPushButton(PottsDlg)

        self.cancelPB.setObjectName("cancelPB")

        self.horizontalLayout.addWidget(self.cancelPB)

        self.horizontalLayout_2.addLayout(self.horizontalLayout)

        self.verticalLayout_2.addLayout(self.horizontalLayout_2)



        self.retranslateUi(PottsDlg)

        self.okPB.clicked.connect(PottsDlg.accept)

        self.cancelPB.clicked.connect(PottsDlg.reject)

        QtCore.QMetaObject.connectSlotsByName(PottsDlg)



    def retranslateUi(self, PottsDlg):

        _translate = QtCore.QCoreApplication.translate

        PottsDlg.setWindowTitle(_translate("PottsDlg", "Potts Configuration"))

        self.groupBox_7.setTitle(_translate("PottsDlg", "Lattice Dimensions"))

        self.label_28.setText(_translate("PottsDlg", "x"))

        self.label_29.setText(_translate("PottsDlg", "y"))

        self.label_30.setText(_translate("PottsDlg", "z"))

        self.groupBox_3.setTitle(_translate("PottsDlg", "Boundary Conditions (Cell Lattice)"))

        self.label_13.setText(_translate("PottsDlg", "x"))

        self.xbcCB.setItemText(0, _translate("PottsDlg", "NoFlux"))

        self.xbcCB.setItemText(1, _translate("PottsDlg", "Periodic"))

        self.label_14.setText(_translate("PottsDlg", "y"))

        self.ybcCB.setItemText(0, _translate("PottsDlg", "NoFlux"))

        self.ybcCB.setItemText(1, _translate("PottsDlg", "Periodic"))

        self.label_15.setText(_translate("PottsDlg", "z"))

        self.zbcCB.setItemText(0, _translate("PottsDlg", "NoFlux"))

        self.zbcCB.setItemText(1, _translate("PottsDlg", "Periodic"))

        self.label_44.setText(_translate("PottsDlg", "LatticeType"))

        self.latticeTypeCB.setItemText(0, _translate("PottsDlg", "Square"))

        self.latticeTypeCB.setItemText(1, _translate("PottsDlg", "Hexagonal"))

        self.label_31.setText(_translate("PottsDlg", "Average Membrane Fluctuations"))

        self.membraneFluctuationsLE.setToolTip(_translate("PottsDlg", "Also known as so called temperature parameter in the acceptance probability expresion:  exp(-delta E/(k*T))"))

        self.membraneFluctuationsLE.setText(_translate("PottsDlg", "10"))

        self.label_32.setText(_translate("PottsDlg", "Pixel Copy Range (NeighborOrder)"))

        self.label_33.setText(_translate("PottsDlg", "Number of MC Steps"))

        self.label_34.setText(_translate("PottsDlg", "Anneal - num steps"))

        self.auto_gen_rand_seed_CB.setText(_translate("PottsDlg", "Autogenerate Random Seed"))

        self.okPB.setText(_translate("PottsDlg", "OK"))

        self.cancelPB.setText(_translate("PottsDlg", "Cancel"))



