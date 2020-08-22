# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'gotolinedlg.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_GoToLineDlg(object):

    def setupUi(self, GoToLineDlg):

        GoToLineDlg.setObjectName("GoToLineDlg")

        GoToLineDlg.resize(337, 117)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(GoToLineDlg)

        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.verticalLayout = QtWidgets.QVBoxLayout()

        self.verticalLayout.setObjectName("verticalLayout")

        self.horizontalLayout = QtWidgets.QHBoxLayout()

        self.horizontalLayout.setObjectName("horizontalLayout")

        self.goToLineLabel = QtWidgets.QLabel(GoToLineDlg)

        self.goToLineLabel.setObjectName("goToLineLabel")

        self.horizontalLayout.addWidget(self.goToLineLabel)

        self.goToLineEdit = QtWidgets.QLineEdit(GoToLineDlg)

        self.goToLineEdit.setObjectName("goToLineEdit")

        self.horizontalLayout.addWidget(self.goToLineEdit)

        self.verticalLayout.addLayout(self.horizontalLayout)

        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.verticalLayout.addItem(spacerItem)

        self.horizontalLayout_2.addLayout(self.verticalLayout)

        self.line = QtWidgets.QFrame(GoToLineDlg)

        self.line.setFrameShape(QtWidgets.QFrame.VLine)

        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.line.setObjectName("line")

        self.horizontalLayout_2.addWidget(self.line)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()

        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.goButton = QtWidgets.QPushButton(GoToLineDlg)

        self.goButton.setObjectName("goButton")

        self.verticalLayout_2.addWidget(self.goButton)

        spacerItem1 = QtWidgets.QSpacerItem(17, 13, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(spacerItem1)

        self.closeButton = QtWidgets.QPushButton(GoToLineDlg)

        self.closeButton.setObjectName("closeButton")

        self.verticalLayout_2.addWidget(self.closeButton)

        self.horizontalLayout_2.addLayout(self.verticalLayout_2)

        self.goToLineLabel.setBuddy(self.goToLineEdit)



        self.retranslateUi(GoToLineDlg)

        self.closeButton.clicked.connect(GoToLineDlg.reject)

        QtCore.QMetaObject.connectSlotsByName(GoToLineDlg)



    def retranslateUi(self, GoToLineDlg):

        _translate = QtCore.QCoreApplication.translate

        GoToLineDlg.setWindowTitle(_translate("GoToLineDlg", "Go To Line"))

        self.goToLineLabel.setText(_translate("GoToLineDlg", "Go to line..."))

        self.goButton.setText(_translate("GoToLineDlg", "Go"))

        self.closeButton.setText(_translate("GoToLineDlg", "Close"))



