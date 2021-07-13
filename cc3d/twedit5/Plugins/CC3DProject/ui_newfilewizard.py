# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'NewFileWizard.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_NewFileWizard(object):

    def setupUi(self, NewFileWizard):

        NewFileWizard.setObjectName("NewFileWizard")

        NewFileWizard.resize(376, 304)

        self.wizardPage1 = QtWidgets.QWizardPage()

        self.wizardPage1.setObjectName("wizardPage1")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.wizardPage1)

        self.verticalLayout.setObjectName("verticalLayout")

        self.gridLayout = QtWidgets.QGridLayout()

        self.gridLayout.setObjectName("gridLayout")

        self.label_2 = QtWidgets.QLabel(self.wizardPage1)

        self.label_2.setObjectName("label_2")

        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout.addItem(spacerItem, 0, 1, 1, 2)

        self.nameLE = QtWidgets.QLineEdit(self.wizardPage1)

        self.nameLE.setObjectName("nameLE")

        self.gridLayout.addWidget(self.nameLE, 0, 3, 1, 2)

        self.nameBrowsePB = QtWidgets.QPushButton(self.wizardPage1)

        self.nameBrowsePB.setObjectName("nameBrowsePB")

        self.gridLayout.addWidget(self.nameBrowsePB, 0, 5, 1, 1)

        self.label = QtWidgets.QLabel(self.wizardPage1)

        self.label.setObjectName("label")

        self.gridLayout.addWidget(self.label, 1, 0, 1, 2)

        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout.addItem(spacerItem1, 1, 2, 3, 1)

        self.fileTypeCB = QtWidgets.QComboBox(self.wizardPage1)

        self.fileTypeCB.setObjectName("fileTypeCB")

        self.fileTypeCB.addItem("")

        self.fileTypeCB.addItem("")

        self.fileTypeCB.addItem("")

        self.fileTypeCB.addItem("")

        self.fileTypeCB.addItem("")

        self.gridLayout.addWidget(self.fileTypeCB, 1, 3, 2, 2)

        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout.addItem(spacerItem2, 1, 5, 1, 1)

        self.label_4 = QtWidgets.QLabel(self.wizardPage1)

        self.label_4.setObjectName("label_4")

        self.gridLayout.addWidget(self.label_4, 2, 0, 2, 2)

        self.locationBrowsePB = QtWidgets.QPushButton(self.wizardPage1)

        self.locationBrowsePB.setObjectName("locationBrowsePB")

        self.gridLayout.addWidget(self.locationBrowsePB, 2, 5, 2, 1)

        self.locationLE = QtWidgets.QLineEdit(self.wizardPage1)

        self.locationLE.setObjectName("locationLE")

        self.gridLayout.addWidget(self.locationLE, 3, 3, 1, 2)

        self.label_3 = QtWidgets.QLabel(self.wizardPage1)

        self.label_3.setObjectName("label_3")

        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 3)

        self.projectDirLE = QtWidgets.QLineEdit(self.wizardPage1)

        self.projectDirLE.setReadOnly(False)

        self.projectDirLE.setObjectName("projectDirLE")

        self.gridLayout.addWidget(self.projectDirLE, 4, 3, 1, 2)

        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout.addItem(spacerItem3, 4, 5, 1, 1)

        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout.addItem(spacerItem4, 5, 3, 2, 2)

        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout.addItem(spacerItem5, 5, 5, 3, 1)

        self.customTypeLE = QtWidgets.QLineEdit(self.wizardPage1)

        self.customTypeLE.setEnabled(False)

        self.customTypeLE.setObjectName("customTypeLE")

        self.gridLayout.addWidget(self.customTypeLE, 6, 4, 2, 1)

        self.customTypeCHB = QtWidgets.QCheckBox(self.wizardPage1)

        self.customTypeCHB.setObjectName("customTypeCHB")

        self.gridLayout.addWidget(self.customTypeCHB, 7, 0, 1, 3)

        self.verticalLayout.addLayout(self.gridLayout)

        spacerItem6 = QtWidgets.QSpacerItem(20, 71, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.verticalLayout.addItem(spacerItem6)

        NewFileWizard.addPage(self.wizardPage1)



        self.retranslateUi(NewFileWizard)

        self.customTypeCHB.toggled['bool'].connect(self.customTypeLE.setEnabled)

        self.customTypeCHB.toggled['bool'].connect(self.fileTypeCB.setDisabled)

        QtCore.QMetaObject.connectSlotsByName(NewFileWizard)

        NewFileWizard.setTabOrder(self.nameLE, self.nameBrowsePB)

        NewFileWizard.setTabOrder(self.nameBrowsePB, self.locationLE)

        NewFileWizard.setTabOrder(self.locationLE, self.locationBrowsePB)

        NewFileWizard.setTabOrder(self.locationBrowsePB, self.fileTypeCB)

        NewFileWizard.setTabOrder(self.fileTypeCB, self.customTypeCHB)

        NewFileWizard.setTabOrder(self.customTypeCHB, self.customTypeLE)

        NewFileWizard.setTabOrder(self.customTypeLE, self.projectDirLE)



    def retranslateUi(self, NewFileWizard):

        _translate = QtCore.QCoreApplication.translate

        NewFileWizard.setWindowTitle(_translate("NewFileWizard", "Add New File (Resource) to CC3D Simulation"))

        self.label_2.setText(_translate("NewFileWizard", "Name:"))

        self.nameBrowsePB.setText(_translate("NewFileWizard", "Browse..."))

        self.label.setText(_translate("NewFileWizard", "File Type:"))

        self.fileTypeCB.setItemText(0, _translate("NewFileWizard", "Main Python Script"))

        self.fileTypeCB.setItemText(1, _translate("NewFileWizard", "XML Script"))

        self.fileTypeCB.setItemText(2, _translate("NewFileWizard", "Python File"))

        self.fileTypeCB.setItemText(3, _translate("NewFileWizard", "PIF File"))

        self.fileTypeCB.setItemText(4, _translate("NewFileWizard", "Concentration File"))

        self.label_4.setText(_translate("NewFileWizard", " Location:"))

        self.locationBrowsePB.setText(_translate("NewFileWizard", "Browse..."))

        self.label_3.setText(_translate("NewFileWizard", "Project Directory"))

        self.customTypeCHB.setText(_translate("NewFileWizard", " Custom File Type "))



