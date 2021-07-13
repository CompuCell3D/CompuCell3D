# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sbmlloaddlg.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SBMLLoadDlg(object):
    def setupUi(self, SBMLLoadDlg):
        SBMLLoadDlg.setObjectName("SBMLLoadDlg")
        SBMLLoadDlg.resize(442, 146)
        self.verticalLayout = QtWidgets.QVBoxLayout(SBMLLoadDlg)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(SBMLLoadDlg)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.fileNameLE = QtWidgets.QLineEdit(SBMLLoadDlg)
        self.fileNameLE.setObjectName("fileNameLE")
        self.gridLayout.addWidget(self.fileNameLE, 0, 1, 1, 1)
        self.browsePB = QtWidgets.QPushButton(SBMLLoadDlg)
        self.browsePB.setObjectName("browsePB")
        self.gridLayout.addWidget(self.browsePB, 0, 2, 1, 1)
        self.label = QtWidgets.QLabel(SBMLLoadDlg)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.modelNameLE = QtWidgets.QLineEdit(SBMLLoadDlg)
        self.modelNameLE.setObjectName("modelNameLE")
        self.gridLayout.addWidget(self.modelNameLE, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(SBMLLoadDlg)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.modelNicknameLE = QtWidgets.QLineEdit(SBMLLoadDlg)
        self.modelNicknameLE.setObjectName("modelNicknameLE")
        self.gridLayout.addWidget(self.modelNicknameLE, 2, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.okPB = QtWidgets.QPushButton(SBMLLoadDlg)
        self.okPB.setObjectName("okPB")
        self.horizontalLayout.addWidget(self.okPB)
        self.leaveEmptyPB = QtWidgets.QPushButton(SBMLLoadDlg)
        self.leaveEmptyPB.setObjectName("leaveEmptyPB")
        self.horizontalLayout.addWidget(self.leaveEmptyPB)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(SBMLLoadDlg)
        self.okPB.clicked.connect(SBMLLoadDlg.accept)
        self.leaveEmptyPB.clicked.connect(SBMLLoadDlg.reject)
        QtCore.QMetaObject.connectSlotsByName(SBMLLoadDlg)

    def retranslateUi(self, SBMLLoadDlg):
        _translate = QtCore.QCoreApplication.translate
        SBMLLoadDlg.setWindowTitle(_translate("SBMLLoadDlg", "Configure SBML file data"))
        self.label_3.setText(_translate("SBMLLoadDlg", "Path to SBML Model"))
        self.browsePB.setText(_translate("SBMLLoadDlg", "Browse..."))
        self.label.setText(_translate("SBMLLoadDlg", "Model Name"))
        self.label_2.setText(_translate("SBMLLoadDlg", "Model Nickame"))
        self.modelNicknameLE.setToolTip(_translate("SBMLLoadDlg", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This is usually abreviated version of Model Name but can be the same as Model Name. E.g. <span style=\" font-weight:600;\">DeltaNotch</span> model can be abreviated <span style=\" font-weight:600;\">DN</span></p></body></html>"))
        self.okPB.setText(_translate("SBMLLoadDlg", "OK"))
        self.leaveEmptyPB.setText(_translate("SBMLLoadDlg", "Leave Empty"))

