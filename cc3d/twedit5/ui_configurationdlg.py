# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'configurationdlg.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_ConfigurationDlg(object):

    def setupUi(self, ConfigurationDlg):

        ConfigurationDlg.setObjectName("ConfigurationDlg")

        ConfigurationDlg.resize(530, 446)

        self.verticalLayout_3 = QtWidgets.QVBoxLayout(ConfigurationDlg)

        self.verticalLayout_3.setObjectName("verticalLayout_3")

        self.tabWidget = QtWidgets.QTabWidget(ConfigurationDlg)

        self.tabWidget.setObjectName("tabWidget")

        self.editingTab = QtWidgets.QWidget()

        self.editingTab.setObjectName("editingTab")

        self.gridLayout_2 = QtWidgets.QGridLayout(self.editingTab)

        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)

        self.gridLayout_2.setObjectName("gridLayout_2")

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()

        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.gridLayout = QtWidgets.QGridLayout()

        self.gridLayout.setObjectName("gridLayout")

        self.tabSpacesCheckBox = QtWidgets.QCheckBox(self.editingTab)

        self.tabSpacesCheckBox.setChecked(True)

        self.tabSpacesCheckBox.setObjectName("tabSpacesCheckBox")

        self.gridLayout.addWidget(self.tabSpacesCheckBox, 0, 0, 1, 3)

        self.tabWidthLabel = QtWidgets.QLabel(self.editingTab)

        self.tabWidthLabel.setObjectName("tabWidthLabel")

        self.gridLayout.addWidget(self.tabWidthLabel, 1, 0, 1, 1)

        self.spacesSpinBox = QtWidgets.QSpinBox(self.editingTab)

        self.spacesSpinBox.setEnabled(True)

        self.spacesSpinBox.setMinimum(1)

        self.spacesSpinBox.setProperty("value", 4)

        self.spacesSpinBox.setObjectName("spacesSpinBox")

        self.gridLayout.addWidget(self.spacesSpinBox, 1, 1, 1, 1)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout.addItem(spacerItem, 1, 2, 1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout)

        self.verticalLayout = QtWidgets.QVBoxLayout()

        self.verticalLayout.setObjectName("verticalLayout")

        self.lineNumberCheckBox = QtWidgets.QCheckBox(self.editingTab)

        self.lineNumberCheckBox.setChecked(True)

        self.lineNumberCheckBox.setObjectName("lineNumberCheckBox")

        self.verticalLayout.addWidget(self.lineNumberCheckBox)

        self.foldTextCheckBox = QtWidgets.QCheckBox(self.editingTab)

        self.foldTextCheckBox.setChecked(True)

        self.foldTextCheckBox.setObjectName("foldTextCheckBox")

        self.verticalLayout.addWidget(self.foldTextCheckBox)

        self.tabGuidelinesCheckBox = QtWidgets.QCheckBox(self.editingTab)

        self.tabGuidelinesCheckBox.setChecked(True)

        self.tabGuidelinesCheckBox.setObjectName("tabGuidelinesCheckBox")

        self.verticalLayout.addWidget(self.tabGuidelinesCheckBox)

        self.whiteSpaceCheckBox = QtWidgets.QCheckBox(self.editingTab)

        self.whiteSpaceCheckBox.setObjectName("whiteSpaceCheckBox")

        self.verticalLayout.addWidget(self.whiteSpaceCheckBox)

        self.eolCheckBox = QtWidgets.QCheckBox(self.editingTab)

        self.eolCheckBox.setObjectName("eolCheckBox")

        self.verticalLayout.addWidget(self.eolCheckBox)

        self.restoreTabsCheckBox = QtWidgets.QCheckBox(self.editingTab)

        self.restoreTabsCheckBox.setChecked(True)

        self.restoreTabsCheckBox.setObjectName("restoreTabsCheckBox")

        self.verticalLayout.addWidget(self.restoreTabsCheckBox)

        self.wrapLinesCheckBox = QtWidgets.QCheckBox(self.editingTab)

        self.wrapLinesCheckBox.setObjectName("wrapLinesCheckBox")

        self.verticalLayout.addWidget(self.wrapLinesCheckBox)

        self.showWrapSymbolCheckBox = QtWidgets.QCheckBox(self.editingTab)

        self.showWrapSymbolCheckBox.setObjectName("showWrapSymbolCheckBox")

        self.verticalLayout.addWidget(self.showWrapSymbolCheckBox)

        self.autocompletionCheckBox = QtWidgets.QCheckBox(self.editingTab)

        self.autocompletionCheckBox.setChecked(True)

        self.autocompletionCheckBox.setObjectName("autocompletionCheckBox")

        self.verticalLayout.addWidget(self.autocompletionCheckBox)

        self.quickTextDecodingCB = QtWidgets.QCheckBox(self.editingTab)

        self.quickTextDecodingCB.setChecked(True)

        self.quickTextDecodingCB.setObjectName("quickTextDecodingCB")

        self.verticalLayout.addWidget(self.quickTextDecodingCB)

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_3.setObjectName("horizontalLayout_3")

        self.autocompletionLabel = QtWidgets.QLabel(self.editingTab)

        self.autocompletionLabel.setObjectName("autocompletionLabel")

        self.horizontalLayout_3.addWidget(self.autocompletionLabel)

        self.autocompletionSpinBox = QtWidgets.QSpinBox(self.editingTab)

        self.autocompletionSpinBox.setMinimum(1)

        self.autocompletionSpinBox.setProperty("value", 2)

        self.autocompletionSpinBox.setObjectName("autocompletionSpinBox")

        self.horizontalLayout_3.addWidget(self.autocompletionSpinBox)

        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.gridLayout_2.addLayout(self.verticalLayout_2, 0, 0, 1, 1)

        spacerItem1 = QtWidgets.QSpacerItem(204, 291, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout_2.addItem(spacerItem1, 0, 1, 1, 1)

        spacerItem2 = QtWidgets.QSpacerItem(177, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.gridLayout_2.addItem(spacerItem2, 1, 0, 1, 1)

        self.tabWidget.addTab(self.editingTab, "")

        self.tab_2 = QtWidgets.QWidget()

        self.tab_2.setObjectName("tab_2")

        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.tab_2)

        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout_9.setObjectName("verticalLayout_9")

        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_8.setObjectName("horizontalLayout_8")

        self.verticalLayout_8 = QtWidgets.QVBoxLayout()

        self.verticalLayout_8.setObjectName("verticalLayout_8")

        self.groupBox = QtWidgets.QGroupBox(self.tab_2)

        self.groupBox.setObjectName("groupBox")

        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox)

        self.verticalLayout_5.setObjectName("verticalLayout_5")

        self.themeCB = QtWidgets.QComboBox(self.groupBox)

        self.themeCB.setObjectName("themeCB")

        self.verticalLayout_5.addWidget(self.themeCB)

        self.verticalLayout_8.addWidget(self.groupBox)

        self.fontGroupBox = QtWidgets.QGroupBox(self.tab_2)

        self.fontGroupBox.setObjectName("fontGroupBox")

        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.fontGroupBox)

        self.verticalLayout_7.setObjectName("verticalLayout_7")

        self.verticalLayout_4 = QtWidgets.QVBoxLayout()

        self.verticalLayout_4.setObjectName("verticalLayout_4")

        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        self.fontLabel = QtWidgets.QLabel(self.fontGroupBox)

        self.fontLabel.setObjectName("fontLabel")

        self.horizontalLayout_4.addWidget(self.fontLabel)

        self.fontComboBox = QtWidgets.QFontComboBox(self.fontGroupBox)

        font = QtGui.QFont()

        font.setFamily("Courier New")

        font.setPointSize(10)

        self.fontComboBox.setCurrentFont(font)

        self.fontComboBox.setObjectName("fontComboBox")

        self.horizontalLayout_4.addWidget(self.fontComboBox)

        self.verticalLayout_4.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_5.setObjectName("horizontalLayout_5")

        self.fontSizeLabel = QtWidgets.QLabel(self.fontGroupBox)

        self.fontSizeLabel.setObjectName("fontSizeLabel")

        self.horizontalLayout_5.addWidget(self.fontSizeLabel)

        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(spacerItem3)

        self.fontSizeComboBox = QtWidgets.QComboBox(self.fontGroupBox)

        self.fontSizeComboBox.setObjectName("fontSizeComboBox")

        self.fontSizeComboBox.addItem("")

        self.fontSizeComboBox.addItem("")

        self.fontSizeComboBox.addItem("")

        self.fontSizeComboBox.addItem("")

        self.fontSizeComboBox.addItem("")

        self.fontSizeComboBox.addItem("")

        self.fontSizeComboBox.addItem("")

        self.fontSizeComboBox.addItem("")

        self.fontSizeComboBox.addItem("")

        self.fontSizeComboBox.addItem("")

        self.fontSizeComboBox.addItem("")

        self.fontSizeComboBox.addItem("")

        self.fontSizeComboBox.addItem("")

        self.horizontalLayout_5.addWidget(self.fontSizeComboBox)

        self.verticalLayout_4.addLayout(self.horizontalLayout_5)

        self.verticalLayout_7.addLayout(self.verticalLayout_4)

        self.verticalLayout_8.addWidget(self.fontGroupBox)

        self.horizontalLayout_8.addLayout(self.verticalLayout_8)

        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout_8.addItem(spacerItem4)

        self.verticalLayout_9.addLayout(self.horizontalLayout_8)

        spacerItem5 = QtWidgets.QSpacerItem(17, 165, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.verticalLayout_9.addItem(spacerItem5)

        self.tabWidget.addTab(self.tab_2, "")

        self.tab = QtWidgets.QWidget()

        self.tab.setObjectName("tab")

        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.tab)

        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout_10.setObjectName("verticalLayout_10")

        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_7.setObjectName("horizontalLayout_7")

        self.pluginsLW = QtWidgets.QListWidget(self.tab)

        self.pluginsLW.setObjectName("pluginsLW")

        self.horizontalLayout_7.addWidget(self.pluginsLW)

        self.verticalLayout_6 = QtWidgets.QVBoxLayout()

        self.verticalLayout_6.setObjectName("verticalLayout_6")

        self.pluginsTE = QtWidgets.QTextEdit(self.tab)

        self.pluginsTE.setReadOnly(True)

        self.pluginsTE.setObjectName("pluginsTE")

        self.verticalLayout_6.addWidget(self.pluginsTE)

        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_6.setObjectName("horizontalLayout_6")

        self.loadPB = QtWidgets.QPushButton(self.tab)

        self.loadPB.setObjectName("loadPB")

        self.horizontalLayout_6.addWidget(self.loadPB)

        self.unloadPB = QtWidgets.QPushButton(self.tab)

        self.unloadPB.setObjectName("unloadPB")

        self.horizontalLayout_6.addWidget(self.unloadPB)

        self.loadOnStartupCHB = QtWidgets.QCheckBox(self.tab)

        self.loadOnStartupCHB.setObjectName("loadOnStartupCHB")

        self.horizontalLayout_6.addWidget(self.loadOnStartupCHB)

        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(spacerItem6)

        self.verticalLayout_6.addLayout(self.horizontalLayout_6)

        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.verticalLayout_6.addItem(spacerItem7)

        self.horizontalLayout_7.addLayout(self.verticalLayout_6)

        self.verticalLayout_10.addLayout(self.horizontalLayout_7)

        self.tabWidget.addTab(self.tab, "")

        self.verticalLayout_3.addWidget(self.tabWidget)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(spacerItem8)

        self.horizontalLayout = QtWidgets.QHBoxLayout()

        self.horizontalLayout.setObjectName("horizontalLayout")

        self.okButton = QtWidgets.QPushButton(ConfigurationDlg)

        self.okButton.setObjectName("okButton")

        self.horizontalLayout.addWidget(self.okButton)

        self.cancelButton = QtWidgets.QPushButton(ConfigurationDlg)

        self.cancelButton.setObjectName("cancelButton")

        self.horizontalLayout.addWidget(self.cancelButton)

        self.horizontalLayout_2.addLayout(self.horizontalLayout)

        self.verticalLayout_3.addLayout(self.horizontalLayout_2)



        self.retranslateUi(ConfigurationDlg)

        self.tabWidget.setCurrentIndex(0)

        self.fontSizeComboBox.setCurrentIndex(2)

        self.tabSpacesCheckBox.toggled['bool'].connect(self.spacesSpinBox.setEnabled)

        self.cancelButton.clicked.connect(ConfigurationDlg.reject)

        self.okButton.clicked.connect(ConfigurationDlg.accept)

        self.wrapLinesCheckBox.toggled['bool'].connect(self.showWrapSymbolCheckBox.setEnabled)

        self.autocompletionCheckBox.toggled['bool'].connect(self.autocompletionSpinBox.setEnabled)

        QtCore.QMetaObject.connectSlotsByName(ConfigurationDlg)



    def retranslateUi(self, ConfigurationDlg):

        _translate = QtCore.QCoreApplication.translate

        ConfigurationDlg.setWindowTitle(_translate("ConfigurationDlg", "Configuration"))

        self.tabSpacesCheckBox.setToolTip(_translate("ConfigurationDlg", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"

"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"

"p, li { white-space: pre-wrap; }\n"

"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"

"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Set this option if you want to  use multiple spaces when Tab key is pressed instead of </span><span style=\" font-size:8pt; font-weight:600;\">&quot;\\t&quot;</span><span style=\" font-size:8pt;\"> character.</span></p>\n"

"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">This is highly recommended when editing Python scripts (unwritten standard is 4 spaces per tab)</span></p></body></html>"))

        self.tabSpacesCheckBox.setText(_translate("ConfigurationDlg", "Use spaces instead of tabs"))

        self.tabWidthLabel.setText(_translate("ConfigurationDlg", "Tab width"))

        self.lineNumberCheckBox.setText(_translate("ConfigurationDlg", "Display Line Number"))

        self.foldTextCheckBox.setText(_translate("ConfigurationDlg", "Enable Text Folding"))

        self.tabGuidelinesCheckBox.setToolTip(_translate("ConfigurationDlg", "Display/hide vertical marks that help visualize indentation"))

        self.tabGuidelinesCheckBox.setText(_translate("ConfigurationDlg", "Enable Tab Guidelines"))

        self.whiteSpaceCheckBox.setToolTip(_translate("ConfigurationDlg", "Display/hide normally invisible white spaces"))

        self.whiteSpaceCheckBox.setText(_translate("ConfigurationDlg", "Display whitespaces"))

        self.eolCheckBox.setText(_translate("ConfigurationDlg", "Display End of Line"))

        self.restoreTabsCheckBox.setText(_translate("ConfigurationDlg", "Restore tabs on startup"))

        self.wrapLinesCheckBox.setText(_translate("ConfigurationDlg", "Wrap Lines"))

        self.showWrapSymbolCheckBox.setText(_translate("ConfigurationDlg", "Show Wrap Symbol"))

        self.autocompletionCheckBox.setText(_translate("ConfigurationDlg", "Enable Autocompletion"))

        self.quickTextDecodingCB.setToolTip(_translate("ConfigurationDlg", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"

"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"

"p, li { white-space: pre-wrap; }\n"

"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"

"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Enales text decoding to be based on first 1000 characters. </span></p>\n"

"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">It can be less acurate then full text scanning but documents open faster </span></p></body></html>"))

        self.quickTextDecodingCB.setText(_translate("ConfigurationDlg", "Enable Quick Text Decoding"))

        self.autocompletionLabel.setToolTip(_translate("ConfigurationDlg", "Determines minimum number of characters after which autocompletion options will be displayed"))

        self.autocompletionLabel.setText(_translate("ConfigurationDlg", "Autocompletion threshold"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.editingTab), _translate("ConfigurationDlg", "Editing"))

        self.groupBox.setTitle(_translate("ConfigurationDlg", "Editor Theme"))

        self.fontGroupBox.setTitle(_translate("ConfigurationDlg", "Fonts"))

        self.fontLabel.setText(_translate("ConfigurationDlg", "Font"))

        self.fontSizeLabel.setText(_translate("ConfigurationDlg", "Size"))

        self.fontSizeComboBox.setItemText(0, _translate("ConfigurationDlg", "8"))

        self.fontSizeComboBox.setItemText(1, _translate("ConfigurationDlg", "9"))

        self.fontSizeComboBox.setItemText(2, _translate("ConfigurationDlg", "10"))

        self.fontSizeComboBox.setItemText(3, _translate("ConfigurationDlg", "11"))

        self.fontSizeComboBox.setItemText(4, _translate("ConfigurationDlg", "12"))

        self.fontSizeComboBox.setItemText(5, _translate("ConfigurationDlg", "14"))

        self.fontSizeComboBox.setItemText(6, _translate("ConfigurationDlg", "16"))

        self.fontSizeComboBox.setItemText(7, _translate("ConfigurationDlg", "18"))

        self.fontSizeComboBox.setItemText(8, _translate("ConfigurationDlg", "20"))

        self.fontSizeComboBox.setItemText(9, _translate("ConfigurationDlg", "22"))

        self.fontSizeComboBox.setItemText(10, _translate("ConfigurationDlg", "24"))

        self.fontSizeComboBox.setItemText(11, _translate("ConfigurationDlg", "26"))

        self.fontSizeComboBox.setItemText(12, _translate("ConfigurationDlg", "28"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("ConfigurationDlg", "Style Config"))

        self.loadPB.setText(_translate("ConfigurationDlg", "Load"))

        self.unloadPB.setText(_translate("ConfigurationDlg", "Unload"))

        self.loadOnStartupCHB.setText(_translate("ConfigurationDlg", "Load on startup"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("ConfigurationDlg", "Plugins"))

        self.okButton.setText(_translate("ConfigurationDlg", "OK"))

        self.cancelButton.setText(_translate("ConfigurationDlg", "Cancel"))



