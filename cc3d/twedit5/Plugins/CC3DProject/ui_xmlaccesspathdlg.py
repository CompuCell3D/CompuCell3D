# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'xmlaccesspathdlg.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_XMLAccessPathDlg(object):

    def setupUi(self, XMLAccessPathDlg):

        XMLAccessPathDlg.setObjectName("XMLAccessPathDlg")

        XMLAccessPathDlg.resize(630, 276)

        self.verticalLayout = QtWidgets.QVBoxLayout(XMLAccessPathDlg)

        self.verticalLayout.setObjectName("verticalLayout")

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.elementLB = QtWidgets.QLabel(XMLAccessPathDlg)

        self.elementLB.setObjectName("elementLB")

        self.horizontalLayout_2.addWidget(self.elementLB)

        self.elemLE = QtWidgets.QLineEdit(XMLAccessPathDlg)

        self.elemLE.setReadOnly(True)

        self.elemLE.setObjectName("elemLE")

        self.horizontalLayout_2.addWidget(self.elemLE)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.paramTW = QtWidgets.QTableWidget(XMLAccessPathDlg)

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

        self.pushButton = QtWidgets.QPushButton(XMLAccessPathDlg)

        self.pushButton.setObjectName("pushButton")

        self.horizontalLayout.addWidget(self.pushButton)

        self.okPB = QtWidgets.QPushButton(XMLAccessPathDlg)

        self.okPB.setObjectName("okPB")

        self.horizontalLayout.addWidget(self.okPB)

        self.verticalLayout.addLayout(self.horizontalLayout)



        self.retranslateUi(XMLAccessPathDlg)

        self.okPB.clicked.connect(XMLAccessPathDlg.accept)

        self.pushButton.clicked.connect(XMLAccessPathDlg.reject)

        QtCore.QMetaObject.connectSlotsByName(XMLAccessPathDlg)



    def retranslateUi(self, XMLAccessPathDlg):

        _translate = QtCore.QCoreApplication.translate

        XMLAccessPathDlg.setWindowTitle(_translate("XMLAccessPathDlg", "Available XML Components "))

        self.elementLB.setText(_translate("XMLAccessPathDlg", "Element:"))

        item = self.paramTW.horizontalHeaderItem(0)

        item.setText(_translate("XMLAccessPathDlg", "Parameter"))

        item = self.paramTW.horizontalHeaderItem(1)

        item.setText(_translate("XMLAccessPathDlg", "Type"))

        item = self.paramTW.horizontalHeaderItem(2)

        item.setText(_translate("XMLAccessPathDlg", "Value"))

        item = self.paramTW.horizontalHeaderItem(3)

        item.setText(_translate("XMLAccessPathDlg", "Action"))

        self.pushButton.setText(_translate("XMLAccessPathDlg", "Cancel"))

        self.okPB.setText(_translate("XMLAccessPathDlg", "OK"))



from . import CC3DProject_rc

