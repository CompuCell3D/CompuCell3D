# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KeyShortcut.ui'
#
# Created: Sun Jan 23 12:30:07 2011
#      by: PyQt4 UI code generator 4.7.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_KeyShortcutDlg(object):
    def setupUi(self, KeyShortcutDlg):
        KeyShortcutDlg.setObjectName("KeyShortcutDlg")
        KeyShortcutDlg.resize(180, 78)
        self.verticalLayout = QtGui.QVBoxLayout(KeyShortcutDlg)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.grabButton = QtGui.QPushButton(KeyShortcutDlg)
        self.grabButton.setObjectName("grabButton")
        self.horizontalLayout.addWidget(self.grabButton)
        self.keyLabel = QtGui.QLabel(KeyShortcutDlg)
        self.keyLabel.setText("")
        self.keyLabel.setObjectName("keyLabel")
        self.horizontalLayout.addWidget(self.keyLabel)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.okButton = QtGui.QPushButton(KeyShortcutDlg)
        self.okButton.setObjectName("okButton")
        self.horizontalLayout_2.addWidget(self.okButton)
        self.cancelButton = QtGui.QPushButton(KeyShortcutDlg)
        self.cancelButton.setObjectName("cancelButton")
        self.horizontalLayout_2.addWidget(self.cancelButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(KeyShortcutDlg)
        QtCore.QObject.connect(self.cancelButton, QtCore.SIGNAL("clicked()"), KeyShortcutDlg.reject)
        QtCore.QObject.connect(self.okButton, QtCore.SIGNAL("clicked()"), KeyShortcutDlg.accept)
        QtCore.QMetaObject.connectSlotsByName(KeyShortcutDlg)

    def retranslateUi(self, KeyShortcutDlg):
        KeyShortcutDlg.setWindowTitle(QtGui.QApplication.translate("KeyShortcutDlg", "Grab Keyboard Shortcut", None, QtGui.QApplication.UnicodeUTF8))
        self.grabButton.setText(QtGui.QApplication.translate("KeyShortcutDlg", "Grab Shortcut...", None, QtGui.QApplication.UnicodeUTF8))
        self.okButton.setText(QtGui.QApplication.translate("KeyShortcutDlg", "OK", None, QtGui.QApplication.UnicodeUTF8))
        self.cancelButton.setText(QtGui.QApplication.translate("KeyShortcutDlg", "Cancel", None, QtGui.QApplication.UnicodeUTF8))

