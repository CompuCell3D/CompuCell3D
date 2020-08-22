# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'KeyboardShortcuts.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_KeyboardShortcutsDlg(object):

    def setupUi(self, KeyboardShortcutsDlg):

        KeyboardShortcutsDlg.setObjectName("KeyboardShortcutsDlg")

        KeyboardShortcutsDlg.resize(418, 352)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(KeyboardShortcutsDlg)

        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.verticalLayout = QtWidgets.QVBoxLayout()

        self.verticalLayout.setObjectName("verticalLayout")

        self.shortcutTable = QtWidgets.QTableWidget(KeyboardShortcutsDlg)

        self.shortcutTable.setObjectName("shortcutTable")

        self.shortcutTable.setColumnCount(2)

        self.shortcutTable.setRowCount(0)

        item = QtWidgets.QTableWidgetItem()

        self.shortcutTable.setHorizontalHeaderItem(0, item)

        item = QtWidgets.QTableWidgetItem()

        self.shortcutTable.setHorizontalHeaderItem(1, item)

        self.verticalLayout.addWidget(self.shortcutTable)

        self.horizontalLayout = QtWidgets.QHBoxLayout()

        self.horizontalLayout.setObjectName("horizontalLayout")

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout.addItem(spacerItem)

        self.okButton = QtWidgets.QPushButton(KeyboardShortcutsDlg)

        self.okButton.setObjectName("okButton")

        self.horizontalLayout.addWidget(self.okButton)

        self.cancelButton = QtWidgets.QPushButton(KeyboardShortcutsDlg)

        self.cancelButton.setObjectName("cancelButton")

        self.horizontalLayout.addWidget(self.cancelButton)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2.addLayout(self.verticalLayout)



        self.retranslateUi(KeyboardShortcutsDlg)

        self.okButton.clicked.connect(KeyboardShortcutsDlg.accept)

        self.cancelButton.clicked.connect(KeyboardShortcutsDlg.reject)

        QtCore.QMetaObject.connectSlotsByName(KeyboardShortcutsDlg)



    def retranslateUi(self, KeyboardShortcutsDlg):

        _translate = QtCore.QCoreApplication.translate

        KeyboardShortcutsDlg.setWindowTitle(_translate("KeyboardShortcutsDlg", "Assign Keyboard Shortcuts"))

        item = self.shortcutTable.horizontalHeaderItem(0)

        item.setText(_translate("KeyboardShortcutsDlg", "Action"))

        item = self.shortcutTable.horizontalHeaderItem(1)

        item.setText(_translate("KeyboardShortcutsDlg", "Shortcut"))

        self.okButton.setText(_translate("KeyboardShortcutsDlg", "OK"))

        self.cancelButton.setText(_translate("KeyboardShortcutsDlg", "Cancel"))



