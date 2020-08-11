# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'KeyShortcut.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_KeyShortcutDlg(object):

    def setupUi(self, KeyShortcutDlg):

        KeyShortcutDlg.setObjectName("KeyShortcutDlg")

        KeyShortcutDlg.resize(188, 98)

        self.verticalLayout = QtWidgets.QVBoxLayout(KeyShortcutDlg)

        self.verticalLayout.setObjectName("verticalLayout")

        self.horizontalLayout = QtWidgets.QHBoxLayout()

        self.horizontalLayout.setObjectName("horizontalLayout")

        self.grabButton = QtWidgets.QPushButton(KeyShortcutDlg)

        self.grabButton.setObjectName("grabButton")

        self.horizontalLayout.addWidget(self.grabButton)

        self.keyLabel = QtWidgets.QLabel(KeyShortcutDlg)

        self.keyLabel.setText("")

        self.keyLabel.setObjectName("keyLabel")

        self.horizontalLayout.addWidget(self.keyLabel)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.okButton = QtWidgets.QPushButton(KeyShortcutDlg)

        self.okButton.setObjectName("okButton")

        self.horizontalLayout_2.addWidget(self.okButton)

        self.cancelButton = QtWidgets.QPushButton(KeyShortcutDlg)

        self.cancelButton.setObjectName("cancelButton")

        self.horizontalLayout_2.addWidget(self.cancelButton)

        self.verticalLayout.addLayout(self.horizontalLayout_2)



        self.retranslateUi(KeyShortcutDlg)

        self.cancelButton.clicked.connect(KeyShortcutDlg.reject)

        self.okButton.clicked.connect(KeyShortcutDlg.accept)

        QtCore.QMetaObject.connectSlotsByName(KeyShortcutDlg)



    def retranslateUi(self, KeyShortcutDlg):

        _translate = QtCore.QCoreApplication.translate

        KeyShortcutDlg.setWindowTitle(_translate("KeyShortcutDlg", "Grab Keyboard Shortcut"))

        self.grabButton.setText(_translate("KeyShortcutDlg", "Grab Shortcut..."))

        self.okButton.setText(_translate("KeyShortcutDlg", "OK"))

        self.cancelButton.setText(_translate("KeyShortcutDlg", "Cancel"))



