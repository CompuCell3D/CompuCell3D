# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'serializereditdlg.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_SerializerEditDlg(object):

    def setupUi(self, SerializerEditDlg):

        SerializerEditDlg.setObjectName("SerializerEditDlg")

        SerializerEditDlg.resize(216, 206)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(SerializerEditDlg)

        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.enableSerializationCHB = QtWidgets.QCheckBox(SerializerEditDlg)

        self.enableSerializationCHB.setObjectName("enableSerializationCHB")

        self.verticalLayout_2.addWidget(self.enableSerializationCHB)

        self.outputGB = QtWidgets.QGroupBox(SerializerEditDlg)

        self.outputGB.setObjectName("outputGB")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.outputGB)

        self.verticalLayout.setObjectName("verticalLayout")

        self.gridLayout = QtWidgets.QGridLayout()

        self.gridLayout.setObjectName("gridLayout")

        self.label = QtWidgets.QLabel(self.outputGB)

        self.label.setObjectName("label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)

        self.frequencySB = QtWidgets.QSpinBox(self.outputGB)

        self.frequencySB.setMaximum(100000000)

        self.frequencySB.setProperty("value", 100)

        self.frequencySB.setObjectName("frequencySB")

        self.gridLayout.addWidget(self.frequencySB, 0, 2, 1, 1)

        self.label_2 = QtWidgets.QLabel(self.outputGB)

        self.label_2.setObjectName("label_2")

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout.addItem(spacerItem, 1, 1, 1, 1)

        self.fileFormatCB = QtWidgets.QComboBox(self.outputGB)

        self.fileFormatCB.setObjectName("fileFormatCB")

        self.fileFormatCB.addItem("")

        self.fileFormatCB.addItem("")

        self.gridLayout.addWidget(self.fileFormatCB, 1, 2, 1, 1)

        self.verticalLayout.addLayout(self.gridLayout)

        self.multipleDirCHB = QtWidgets.QCheckBox(self.outputGB)

        self.multipleDirCHB.setObjectName("multipleDirCHB")

        self.verticalLayout.addWidget(self.multipleDirCHB)

        self.verticalLayout_2.addWidget(self.outputGB)

        self.enableRestartCHB = QtWidgets.QCheckBox(SerializerEditDlg)

        self.enableRestartCHB.setObjectName("enableRestartCHB")

        self.verticalLayout_2.addWidget(self.enableRestartCHB)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(spacerItem1)

        self.okPB = QtWidgets.QPushButton(SerializerEditDlg)

        self.okPB.setObjectName("okPB")

        self.horizontalLayout_2.addWidget(self.okPB)

        self.cancelPB = QtWidgets.QPushButton(SerializerEditDlg)

        self.cancelPB.setObjectName("cancelPB")

        self.horizontalLayout_2.addWidget(self.cancelPB)

        self.verticalLayout_2.addLayout(self.horizontalLayout_2)



        self.retranslateUi(SerializerEditDlg)

        self.okPB.clicked.connect(SerializerEditDlg.accept)

        self.cancelPB.clicked.connect(SerializerEditDlg.reject)

        self.enableSerializationCHB.toggled['bool'].connect(self.outputGB.setEnabled)

        QtCore.QMetaObject.connectSlotsByName(SerializerEditDlg)



    def retranslateUi(self, SerializerEditDlg):

        _translate = QtCore.QCoreApplication.translate

        SerializerEditDlg.setWindowTitle(_translate("SerializerEditDlg", "Serializer Properties"))

        self.enableSerializationCHB.setText(_translate("SerializerEditDlg", "Allow Serialization"))

        self.outputGB.setTitle(_translate("SerializerEditDlg", "Output Properties"))

        self.label.setText(_translate("SerializerEditDlg", "Output Frequency"))

        self.label_2.setText(_translate("SerializerEditDlg", "File Format"))

        self.fileFormatCB.setItemText(0, _translate("SerializerEditDlg", "text"))

        self.fileFormatCB.setItemText(1, _translate("SerializerEditDlg", "binary"))

        self.multipleDirCHB.setText(_translate("SerializerEditDlg", "Allow multiple restart snapshots"))

        self.enableRestartCHB.setText(_translate("SerializerEditDlg", "Enable Restart"))

        self.okPB.setText(_translate("SerializerEditDlg", "OK"))

        self.cancelPB.setText(_translate("SerializerEditDlg", "Cancel"))



