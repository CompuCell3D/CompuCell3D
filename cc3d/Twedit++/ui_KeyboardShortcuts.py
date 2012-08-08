# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KeyboardShortcuts.ui'
#
# Created: Mon Jan 24 15:27:08 2011
#      by: PyQt4 UI code generator 4.7.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_KeyboardShortcutsDlg(object):
    def setupUi(self, KeyboardShortcutsDlg):
        KeyboardShortcutsDlg.setObjectName("KeyboardShortcutsDlg")
        KeyboardShortcutsDlg.resize(414, 345)
        self.horizontalLayout_2 = QtGui.QHBoxLayout(KeyboardShortcutsDlg)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.shortcutTable = QtGui.QTableWidget(KeyboardShortcutsDlg)
        self.shortcutTable.setObjectName("shortcutTable")
        self.shortcutTable.setColumnCount(2)
        self.shortcutTable.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.shortcutTable.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.shortcutTable.setHorizontalHeaderItem(1, item)
        self.verticalLayout.addWidget(self.shortcutTable)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.okButton = QtGui.QPushButton(KeyboardShortcutsDlg)
        self.okButton.setObjectName("okButton")
        self.horizontalLayout.addWidget(self.okButton)
        self.cancelButton = QtGui.QPushButton(KeyboardShortcutsDlg)
        self.cancelButton.setObjectName("cancelButton")
        self.horizontalLayout.addWidget(self.cancelButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(KeyboardShortcutsDlg)
        QtCore.QObject.connect(self.okButton, QtCore.SIGNAL("clicked()"), KeyboardShortcutsDlg.accept)
        QtCore.QObject.connect(self.cancelButton, QtCore.SIGNAL("clicked()"), KeyboardShortcutsDlg.reject)
        QtCore.QMetaObject.connectSlotsByName(KeyboardShortcutsDlg)

    def retranslateUi(self, KeyboardShortcutsDlg):
        KeyboardShortcutsDlg.setWindowTitle(QtGui.QApplication.translate("KeyboardShortcutsDlg", "Assign Keyboard Shortcuts", None, QtGui.QApplication.UnicodeUTF8))
        self.shortcutTable.horizontalHeaderItem(0).setText(QtGui.QApplication.translate("KeyboardShortcutsDlg", "Action", None, QtGui.QApplication.UnicodeUTF8))
        self.shortcutTable.horizontalHeaderItem(1).setText(QtGui.QApplication.translate("KeyboardShortcutsDlg", "Shortcut", None, QtGui.QApplication.UnicodeUTF8))
        self.okButton.setText(QtGui.QApplication.translate("KeyboardShortcutsDlg", "OK", None, QtGui.QApplication.UnicodeUTF8))
        self.cancelButton.setText(QtGui.QApplication.translate("KeyboardShortcutsDlg", "Cancel", None, QtGui.QApplication.UnicodeUTF8))

