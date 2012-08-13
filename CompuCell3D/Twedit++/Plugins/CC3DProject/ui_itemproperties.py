# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ItemProperties.ui'
#
# Created: Mon Mar 07 12:49:13 2011
#      by: PyQt4 UI code generator 4.7.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_ItemProperties(object):
    def setupUi(self, ItemProperties):
        ItemProperties.setObjectName("ItemProperties")
        ItemProperties.resize(272, 237)
        self.verticalLayout = QtGui.QVBoxLayout(ItemProperties)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtGui.QLabel(ItemProperties)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.pathLabel = QtGui.QLabel(ItemProperties)
        self.pathLabel.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.pathLabel.setWordWrap(False)
        self.pathLabel.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.pathLabel.setObjectName("pathLabel")
        self.gridLayout.addWidget(self.pathLabel, 0, 1, 1, 1)
        self.label_2 = QtGui.QLabel(ItemProperties)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.typeLabel = QtGui.QLabel(ItemProperties)
        self.typeLabel.setObjectName("typeLabel")
        self.gridLayout.addWidget(self.typeLabel, 1, 1, 1, 1)
        self.line = QtGui.QFrame(ItemProperties)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 2, 0, 1, 2)
        self.label_3 = QtGui.QLabel(ItemProperties)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.moduleLE = QtGui.QLineEdit(ItemProperties)
        self.moduleLE.setObjectName("moduleLE")
        self.gridLayout.addWidget(self.moduleLE, 3, 1, 1, 1)
        self.label_4 = QtGui.QLabel(ItemProperties)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 4, 0, 1, 1)
        self.originLE = QtGui.QLineEdit(ItemProperties)
        self.originLE.setObjectName("originLE")
        self.gridLayout.addWidget(self.originLE, 4, 1, 1, 1)
        self.copyCHB = QtGui.QCheckBox(ItemProperties)
        self.copyCHB.setChecked(True)
        self.copyCHB.setObjectName("copyCHB")
        self.gridLayout.addWidget(self.copyCHB, 5, 0, 1, 1)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 5, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        spacerItem1 = QtGui.QSpacerItem(20, 56, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.pushButton = QtGui.QPushButton(ItemProperties)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtGui.QPushButton(ItemProperties)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(ItemProperties)
        QtCore.QObject.connect(self.pushButton, QtCore.SIGNAL("clicked()"), ItemProperties.accept)
        QtCore.QObject.connect(self.pushButton_2, QtCore.SIGNAL("clicked()"), ItemProperties.reject)
        QtCore.QMetaObject.connectSlotsByName(ItemProperties)

    def retranslateUi(self, ItemProperties):
        ItemProperties.setWindowTitle(QtGui.QApplication.translate("ItemProperties", "Item Properties", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("ItemProperties", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600;\">Path</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.pathLabel.setText(QtGui.QApplication.translate("ItemProperties", "TextLabel", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("ItemProperties", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600;\">Type</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.typeLabel.setText(QtGui.QApplication.translate("ItemProperties", "TextLabel", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("ItemProperties", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600;\">Module</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.moduleLE.setToolTip(QtGui.QApplication.translate("ItemProperties", "Name of CC3D module that will be responsible for handling this resource. It has to be the same as the name of plugin/steppable listed in the XML script. E.g. FocalPointPlasticity", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("ItemProperties", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600;\">Origin</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.originLE.setWhatsThis(QtGui.QApplication.translate("ItemProperties", "This is optional label. It informa users whether e.g. a given file was an output of module serialization. This purpose of this lebel is to provide basic provenance information.", None, QtGui.QApplication.UnicodeUTF8))
        self.copyCHB.setToolTip(QtGui.QApplication.translate("ItemProperties", "Checked box means that resource will be copied to simulation directory.", None, QtGui.QApplication.UnicodeUTF8))
        self.copyCHB.setText(QtGui.QApplication.translate("ItemProperties", "Copy", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton.setText(QtGui.QApplication.translate("ItemProperties", "OK", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_2.setText(QtGui.QApplication.translate("ItemProperties", "Cancel", None, QtGui.QApplication.UnicodeUTF8))

