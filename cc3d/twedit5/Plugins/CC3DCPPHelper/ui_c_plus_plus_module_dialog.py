# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'C_Plus_Plus_Module_Dialog.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_C_Plus_Plus_Module_Dialog(object):

    def setupUi(self, C_Plus_Plus_Module_Dialog):

        C_Plus_Plus_Module_Dialog.setObjectName("C_Plus_Plus_Module_Dialog")

        C_Plus_Plus_Module_Dialog.resize(486, 355)

        self.verticalLayout_3 = QtWidgets.QVBoxLayout(C_Plus_Plus_Module_Dialog)

        self.verticalLayout_3.setObjectName("verticalLayout_3")

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()

        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.gridLayout = QtWidgets.QGridLayout()

        self.gridLayout.setObjectName("gridLayout")

        self.label_2 = QtWidgets.QLabel(C_Plus_Plus_Module_Dialog)

        self.label_2.setObjectName("label_2")

        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)

        self.label = QtWidgets.QLabel(C_Plus_Plus_Module_Dialog)

        self.label.setObjectName("label")

        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)

        self.moduleDirLE = QtWidgets.QLineEdit(C_Plus_Plus_Module_Dialog)

        self.moduleDirLE.setObjectName("moduleDirLE")

        self.gridLayout.addWidget(self.moduleDirLE, 1, 1, 1, 1)

        self.moduleDirPB = QtWidgets.QPushButton(C_Plus_Plus_Module_Dialog)

        self.moduleDirPB.setObjectName("moduleDirPB")

        self.gridLayout.addWidget(self.moduleDirPB, 1, 2, 1, 1)

        self.moduleCoreNameLE = QtWidgets.QLineEdit(C_Plus_Plus_Module_Dialog)

        self.moduleCoreNameLE.setObjectName("moduleCoreNameLE")

        self.gridLayout.addWidget(self.moduleCoreNameLE, 0, 1, 1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout)

        self.groupBox = QtWidgets.QGroupBox(C_Plus_Plus_Module_Dialog)

        self.groupBox.setObjectName("groupBox")

        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.groupBox)

        self.horizontalLayout_5.setObjectName("horizontalLayout_5")

        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        self.mainCodeLayoutRB = QtWidgets.QRadioButton(self.groupBox)

        self.mainCodeLayoutRB.setChecked(True)

        self.mainCodeLayoutRB.setObjectName("mainCodeLayoutRB")

        self.horizontalLayout_4.addWidget(self.mainCodeLayoutRB)

        self.developerZoneLayoutRB = QtWidgets.QRadioButton(self.groupBox)

        self.developerZoneLayoutRB.setObjectName("developerZoneLayoutRB")

        self.horizontalLayout_4.addWidget(self.developerZoneLayoutRB)

        self.horizontalLayout_5.addLayout(self.horizontalLayout_4)

        self.verticalLayout_2.addWidget(self.groupBox)

        self.moduleTypeGB = QtWidgets.QGroupBox(C_Plus_Plus_Module_Dialog)

        self.moduleTypeGB.setObjectName("moduleTypeGB")

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.moduleTypeGB)

        self.horizontalLayout_3.setObjectName("horizontalLayout_3")

        self.pluginRB = QtWidgets.QRadioButton(self.moduleTypeGB)

        self.pluginRB.setChecked(True)

        self.pluginRB.setObjectName("pluginRB")

        self.horizontalLayout_3.addWidget(self.pluginRB)

        self.steppableRB = QtWidgets.QRadioButton(self.moduleTypeGB)

        self.steppableRB.setObjectName("steppableRB")

        self.horizontalLayout_3.addWidget(self.steppableRB)

        self.pythonWrapCB = QtWidgets.QCheckBox(self.moduleTypeGB)

        self.pythonWrapCB.setChecked(True)

        self.pythonWrapCB.setObjectName("pythonWrapCB")

        self.horizontalLayout_3.addWidget(self.pythonWrapCB)

        self.extraAttribCB = QtWidgets.QCheckBox(self.moduleTypeGB)

        self.extraAttribCB.setObjectName("extraAttribCB")

        self.horizontalLayout_3.addWidget(self.extraAttribCB)

        self.verticalLayout_2.addWidget(self.moduleTypeGB)

        self.pluginFunctionalityGB = QtWidgets.QGroupBox(C_Plus_Plus_Module_Dialog)

        self.pluginFunctionalityGB.setObjectName("pluginFunctionalityGB")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.pluginFunctionalityGB)

        self.verticalLayout.setObjectName("verticalLayout")

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.energyFcnCB = QtWidgets.QCheckBox(self.pluginFunctionalityGB)

        self.energyFcnCB.setChecked(True)

        self.energyFcnCB.setObjectName("energyFcnCB")

        self.horizontalLayout_2.addWidget(self.energyFcnCB)

        self.latticeMonitorCB = QtWidgets.QCheckBox(self.pluginFunctionalityGB)

        self.latticeMonitorCB.setObjectName("latticeMonitorCB")

        self.horizontalLayout_2.addWidget(self.latticeMonitorCB)

        self.stepperCB = QtWidgets.QCheckBox(self.pluginFunctionalityGB)

        self.stepperCB.setObjectName("stepperCB")

        self.horizontalLayout_2.addWidget(self.stepperCB)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.verticalLayout_2.addWidget(self.pluginFunctionalityGB)

        self.horizontalLayout = QtWidgets.QHBoxLayout()

        self.horizontalLayout.setObjectName("horizontalLayout")

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout.addItem(spacerItem)

        self.okPB = QtWidgets.QPushButton(C_Plus_Plus_Module_Dialog)

        self.okPB.setObjectName("okPB")

        self.horizontalLayout.addWidget(self.okPB)

        self.cancelPB = QtWidgets.QPushButton(C_Plus_Plus_Module_Dialog)

        self.cancelPB.setObjectName("cancelPB")

        self.horizontalLayout.addWidget(self.cancelPB)

        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.verticalLayout_3.addLayout(self.verticalLayout_2)



        self.retranslateUi(C_Plus_Plus_Module_Dialog)

        self.cancelPB.clicked.connect(C_Plus_Plus_Module_Dialog.reject)

        self.steppableRB.toggled['bool'].connect(self.pluginFunctionalityGB.setHidden)

        QtCore.QMetaObject.connectSlotsByName(C_Plus_Plus_Module_Dialog)

        C_Plus_Plus_Module_Dialog.setTabOrder(self.moduleCoreNameLE, self.moduleDirLE)

        C_Plus_Plus_Module_Dialog.setTabOrder(self.moduleDirLE, self.moduleDirPB)

        C_Plus_Plus_Module_Dialog.setTabOrder(self.moduleDirPB, self.pluginRB)

        C_Plus_Plus_Module_Dialog.setTabOrder(self.pluginRB, self.steppableRB)

        C_Plus_Plus_Module_Dialog.setTabOrder(self.steppableRB, self.energyFcnCB)

        C_Plus_Plus_Module_Dialog.setTabOrder(self.energyFcnCB, self.latticeMonitorCB)

        C_Plus_Plus_Module_Dialog.setTabOrder(self.latticeMonitorCB, self.stepperCB)

        C_Plus_Plus_Module_Dialog.setTabOrder(self.stepperCB, self.cancelPB)

        C_Plus_Plus_Module_Dialog.setTabOrder(self.cancelPB, self.okPB)



    def retranslateUi(self, C_Plus_Plus_Module_Dialog):

        _translate = QtCore.QCoreApplication.translate

        C_Plus_Plus_Module_Dialog.setWindowTitle(_translate("C_Plus_Plus_Module_Dialog", "Generate CC3D C++ Module"))

        self.label_2.setText(_translate("C_Plus_Plus_Module_Dialog", "Module Core Name"))

        self.label.setText(_translate("C_Plus_Plus_Module_Dialog", "Module Directory"))

        self.moduleDirPB.setText(_translate("C_Plus_Plus_Module_Dialog", "Browse..."))

        self.groupBox.setTitle(_translate("C_Plus_Plus_Module_Dialog", "Code Layout"))

        self.mainCodeLayoutRB.setText(_translate("C_Plus_Plus_Module_Dialog", "Main Code"))

        self.developerZoneLayoutRB.setText(_translate("C_Plus_Plus_Module_Dialog", "Developer Zone"))

        self.moduleTypeGB.setTitle(_translate("C_Plus_Plus_Module_Dialog", "C++ Module Type"))

        self.pluginRB.setText(_translate("C_Plus_Plus_Module_Dialog", "Plugin"))

        self.steppableRB.setText(_translate("C_Plus_Plus_Module_Dialog", "Steppable"))

        self.pythonWrapCB.setText(_translate("C_Plus_Plus_Module_Dialog", "Python Wrap"))

        self.extraAttribCB.setText(_translate("C_Plus_Plus_Module_Dialog", "Attach cell attribute"))

        self.pluginFunctionalityGB.setTitle(_translate("C_Plus_Plus_Module_Dialog", "Plugin Functionality"))

        self.energyFcnCB.setText(_translate("C_Plus_Plus_Module_Dialog", "EnergyFunction"))

        self.latticeMonitorCB.setText(_translate("C_Plus_Plus_Module_Dialog", "LatticeMonitor"))

        self.stepperCB.setText(_translate("C_Plus_Plus_Module_Dialog", "Stepper"))

        self.okPB.setText(_translate("C_Plus_Plus_Module_Dialog", "OK"))

        self.cancelPB.setText(_translate("C_Plus_Plus_Module_Dialog", "Cancel"))



