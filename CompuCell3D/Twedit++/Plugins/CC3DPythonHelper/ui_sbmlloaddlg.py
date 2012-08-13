# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sbmlloaddlg.ui'
#
# Created: Mon Aug 06 17:37:21 2012
#      by: PyQt4 UI code generator 4.8.6
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_SBMLLoadDlg(object):
    def setupUi(self, SBMLLoadDlg):
        SBMLLoadDlg.setObjectName(_fromUtf8("SBMLLoadDlg"))
        SBMLLoadDlg.resize(442, 146)
        SBMLLoadDlg.setWindowTitle(QtGui.QApplication.translate("SBMLLoadDlg", "Configure SBML file data", None, QtGui.QApplication.UnicodeUTF8))
        self.verticalLayout = QtGui.QVBoxLayout(SBMLLoadDlg)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_3 = QtGui.QLabel(SBMLLoadDlg)
        self.label_3.setText(QtGui.QApplication.translate("SBMLLoadDlg", "Path to SBML Model", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.fileNameLE = QtGui.QLineEdit(SBMLLoadDlg)
        self.fileNameLE.setObjectName(_fromUtf8("fileNameLE"))
        self.gridLayout.addWidget(self.fileNameLE, 0, 1, 1, 1)
        self.browsePB = QtGui.QPushButton(SBMLLoadDlg)
        self.browsePB.setText(QtGui.QApplication.translate("SBMLLoadDlg", "Browse...", None, QtGui.QApplication.UnicodeUTF8))
        self.browsePB.setObjectName(_fromUtf8("browsePB"))
        self.gridLayout.addWidget(self.browsePB, 0, 2, 1, 1)
        self.label = QtGui.QLabel(SBMLLoadDlg)
        self.label.setText(QtGui.QApplication.translate("SBMLLoadDlg", "Model Name", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.modelNameLE = QtGui.QLineEdit(SBMLLoadDlg)
        self.modelNameLE.setObjectName(_fromUtf8("modelNameLE"))
        self.gridLayout.addWidget(self.modelNameLE, 1, 1, 1, 1)
        self.label_2 = QtGui.QLabel(SBMLLoadDlg)
        self.label_2.setText(QtGui.QApplication.translate("SBMLLoadDlg", "Model Nickame", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.modelNicknameLE = QtGui.QLineEdit(SBMLLoadDlg)
        self.modelNicknameLE.setToolTip(QtGui.QApplication.translate("SBMLLoadDlg", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This is usually abreviated version of Model Name but can be the same as Model Name. E.g. <span style=\" font-weight:600;\">DeltaNotch</span> model can be abreviated <span style=\" font-weight:600;\">DN</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.modelNicknameLE.setObjectName(_fromUtf8("modelNicknameLE"))
        self.gridLayout.addWidget(self.modelNicknameLE, 2, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.okPB = QtGui.QPushButton(SBMLLoadDlg)
        self.okPB.setText(QtGui.QApplication.translate("SBMLLoadDlg", "OK", None, QtGui.QApplication.UnicodeUTF8))
        self.okPB.setObjectName(_fromUtf8("okPB"))
        self.horizontalLayout.addWidget(self.okPB)
        self.leaveEmptyPB = QtGui.QPushButton(SBMLLoadDlg)
        self.leaveEmptyPB.setText(QtGui.QApplication.translate("SBMLLoadDlg", "Leave Empty", None, QtGui.QApplication.UnicodeUTF8))
        self.leaveEmptyPB.setObjectName(_fromUtf8("leaveEmptyPB"))
        self.horizontalLayout.addWidget(self.leaveEmptyPB)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(SBMLLoadDlg)
        QtCore.QObject.connect(self.okPB, QtCore.SIGNAL(_fromUtf8("clicked()")), SBMLLoadDlg.accept)
        QtCore.QObject.connect(self.leaveEmptyPB, QtCore.SIGNAL(_fromUtf8("clicked()")), SBMLLoadDlg.reject)
        QtCore.QMetaObject.connectSlotsByName(SBMLLoadDlg)

    def retranslateUi(self, SBMLLoadDlg):
        pass

