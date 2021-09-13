# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_screenshot_description_browser.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_screenshotDescriptionDialog(object):
    def setupUi(self, screenshotDescriptionDialog):
        screenshotDescriptionDialog.setObjectName("screenshotDescriptionDialog")
        screenshotDescriptionDialog.resize(541, 424)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(screenshotDescriptionDialog)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.v_layout = QtWidgets.QVBoxLayout()
        self.v_layout.setObjectName("v_layout")
        self.label = QtWidgets.QLabel(screenshotDescriptionDialog)
        self.label.setObjectName("label")
        self.v_layout.addWidget(self.label)
        self.scr_list_TE = QtWidgets.QPlainTextEdit(screenshotDescriptionDialog)
        self.scr_list_TE.setReadOnly(True)
        self.scr_list_TE.setObjectName("scr_list_TE")
        self.v_layout.addWidget(self.scr_list_TE)
        self.label_2 = QtWidgets.QLabel(screenshotDescriptionDialog)
        self.label_2.setObjectName("label_2")
        self.v_layout.addWidget(self.label_2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.v_layout.addItem(spacerItem)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.clear_screenshots_PB = QtWidgets.QPushButton(screenshotDescriptionDialog)
        self.clear_screenshots_PB.setObjectName("clear_screenshots_PB")
        self.horizontalLayout.addWidget(self.clear_screenshots_PB)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.ok_PB = QtWidgets.QPushButton(screenshotDescriptionDialog)
        self.ok_PB.setObjectName("ok_PB")
        self.horizontalLayout.addWidget(self.ok_PB)
        self.v_layout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.v_layout)

        self.retranslateUi(screenshotDescriptionDialog)
        self.ok_PB.clicked.connect(screenshotDescriptionDialog.accept)
        QtCore.QMetaObject.connectSlotsByName(screenshotDescriptionDialog)

    def retranslateUi(self, screenshotDescriptionDialog):
        _translate = QtCore.QCoreApplication.translate
        screenshotDescriptionDialog.setWindowTitle(_translate("screenshotDescriptionDialog", "Screenshot Description Browser"))
        self.label.setText(_translate("screenshotDescriptionDialog", "Available Screenshot Labels"))
        self.label_2.setText(_translate("screenshotDescriptionDialog", "Screenshot Description "))
        self.clear_screenshots_PB.setText(_translate("screenshotDescriptionDialog", "Clear Screenshots"))
        self.ok_PB.setText(_translate("screenshotDescriptionDialog", "OK"))
