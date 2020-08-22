# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'parameterdlg.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_ParameterDlg(object):

    def setupUi(self, ParameterDlg):

        ParameterDlg.setObjectName("ParameterDlg")

        ParameterDlg.resize(423, 293)

        self.verticalLayout = QtWidgets.QVBoxLayout(ParameterDlg)

        self.verticalLayout.setObjectName("verticalLayout")

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.elementLB = QtWidgets.QLabel(ParameterDlg)

        self.elementLB.setObjectName("elementLB")

        self.horizontalLayout_2.addWidget(self.elementLB)

        self.elemLE = QtWidgets.QLineEdit(ParameterDlg)

        self.elemLE.setReadOnly(True)

        self.elemLE.setObjectName("elemLE")

        self.horizontalLayout_2.addWidget(self.elemLE)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.paramTW = QtWidgets.QTableWidget(ParameterDlg)

        self.paramTW.setObjectName("paramTW")

        self.paramTW.setColumnCount(4)

        self.paramTW.setRowCount(0)

        item = QtWidgets.QTableWidgetItem()

        self.paramTW.setHorizontalHeaderItem(0, item)

        item = QtWidgets.QTableWidgetItem()

        self.paramTW.setHorizontalHeaderItem(1, item)

        item = QtWidgets.QTableWidgetItem()

        self.paramTW.setHorizontalHeaderItem(2, item)

        item = QtWidgets.QTableWidgetItem()

        self.paramTW.setHorizontalHeaderItem(3, item)

        self.verticalLayout.addWidget(self.paramTW)

        self.horizontalLayout = QtWidgets.QHBoxLayout()

        self.horizontalLayout.setObjectName("horizontalLayout")

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout.addItem(spacerItem)

        self.pushButton = QtWidgets.QPushButton(ParameterDlg)

        self.pushButton.setObjectName("pushButton")

        self.horizontalLayout.addWidget(self.pushButton)

        self.okPB = QtWidgets.QPushButton(ParameterDlg)

        self.okPB.setObjectName("okPB")

        self.horizontalLayout.addWidget(self.okPB)

        self.verticalLayout.addLayout(self.horizontalLayout)



        self.retranslateUi(ParameterDlg)

        self.okPB.clicked.connect(ParameterDlg.accept)

        self.pushButton.clicked.connect(ParameterDlg.reject)

        QtCore.QMetaObject.connectSlotsByName(ParameterDlg)



    def retranslateUi(self, ParameterDlg):

        _translate = QtCore.QCoreApplication.translate

        ParameterDlg.setWindowTitle(_translate("ParameterDlg", "Scannable Paramerters"))

        self.elementLB.setText(_translate("ParameterDlg", "Element:"))

        item = self.paramTW.horizontalHeaderItem(0)

        item.setText(_translate("ParameterDlg", "Parameter"))

        item = self.paramTW.horizontalHeaderItem(1)

        item.setText(_translate("ParameterDlg", "Type"))

        item = self.paramTW.horizontalHeaderItem(2)

        item.setText(_translate("ParameterDlg", "Value"))

        item = self.paramTW.horizontalHeaderItem(3)

        item.setText(_translate("ParameterDlg", "Action"))

        self.pushButton.setText(_translate("ParameterDlg", "Cancel"))

        self.okPB.setText(_translate("ParameterDlg", "OK"))



from . import CC3DProject_rc

