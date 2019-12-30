# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'contactdlg.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ContactPluginGUI(object):
    def setupUi(self, ContactPluginGUI):
        ContactPluginGUI.setObjectName("ContactPluginGUI")
        ContactPluginGUI.setWindowModality(QtCore.Qt.NonModal)
        ContactPluginGUI.resize(400, 300)
        self.gridLayout_2 = QtWidgets.QGridLayout(ContactPluginGUI)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(ContactPluginGUI)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.spinBox = QtWidgets.QSpinBox(ContactPluginGUI)
        self.spinBox.setObjectName("spinBox")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.spinBox)
        self.verticalLayout.addLayout(self.formLayout)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.tableWidget = QtWidgets.QTableWidget(ContactPluginGUI)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.horizontalLayout.addWidget(self.tableWidget)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(ContactPluginGUI)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_2.addWidget(self.buttonBox, 1, 0, 1, 1)

        self.retranslateUi(ContactPluginGUI)
        QtCore.QMetaObject.connectSlotsByName(ContactPluginGUI)

    def retranslateUi(self, ContactPluginGUI):
        _translate = QtCore.QCoreApplication.translate
        ContactPluginGUI.setWindowTitle(_translate("ContactPluginGUI", "Contact Plugin: Please define adhesion coefficients"))
        self.label.setText(_translate("ContactPluginGUI", "Neighbor Order"))
