# -*- coding: utf-8 -*-



# Form implementation generated from reading ui file 'findinfiles.ui'

#

# Created by: PyQt5 UI code generator 5.6

#

# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_FindInFiles(object):

    def setupUi(self, FindInFiles):

        FindInFiles.setObjectName("FindInFiles")

        FindInFiles.resize(533, 401)

        FindInFiles.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)

        FindInFiles.setToolTip("")

        FindInFiles.setWhatsThis("")

        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(FindInFiles)

        self.horizontalLayout_7.setObjectName("horizontalLayout_7")

        self.tabWidget = QtWidgets.QTabWidget(FindInFiles)

        self.tabWidget.setObjectName("tabWidget")

        self.tab = QtWidgets.QWidget()

        self.tab.setObjectName("tab")

        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.tab)

        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)

        self.horizontalLayout_12.setObjectName("horizontalLayout_12")

        self.verticalLayout_8 = QtWidgets.QVBoxLayout()

        self.verticalLayout_8.setObjectName("verticalLayout_8")

        self._2 = QtWidgets.QGridLayout()

        self._2.setObjectName("_2")

        self.label_2 = QtWidgets.QLabel(self.tab)

        self.label_2.setObjectName("label_2")

        self._2.addWidget(self.label_2, 1, 0, 1, 1)

        self.label = QtWidgets.QLabel(self.tab)

        self.label.setObjectName("label")

        self._2.addWidget(self.label, 0, 0, 1, 1)

        self.replaceComboBox = QtWidgets.QComboBox(self.tab)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        sizePolicy.setHorizontalStretch(0)

        sizePolicy.setVerticalStretch(0)

        sizePolicy.setHeightForWidth(self.replaceComboBox.sizePolicy().hasHeightForWidth())

        self.replaceComboBox.setSizePolicy(sizePolicy)

        self.replaceComboBox.setMaximumSize(QtCore.QSize(200, 16777215))

        self.replaceComboBox.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)

        self.replaceComboBox.setObjectName("replaceComboBox")

        self._2.addWidget(self.replaceComboBox, 1, 1, 1, 1)

        self.findComboBox = QtWidgets.QComboBox(self.tab)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        sizePolicy.setHorizontalStretch(0)

        sizePolicy.setVerticalStretch(0)

        sizePolicy.setHeightForWidth(self.findComboBox.sizePolicy().hasHeightForWidth())

        self.findComboBox.setSizePolicy(sizePolicy)

        self.findComboBox.setMaximumSize(QtCore.QSize(200, 16777215))

        self.findComboBox.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)

        self.findComboBox.setObjectName("findComboBox")

        self._2.addWidget(self.findComboBox, 0, 1, 1, 1)

        self.verticalLayout_8.addLayout(self._2)

        self._3 = QtWidgets.QHBoxLayout()

        self._3.setObjectName("_3")

        self.wholeCheckBox = QtWidgets.QCheckBox(self.tab)

        self.wholeCheckBox.setChecked(False)

        self.wholeCheckBox.setObjectName("wholeCheckBox")

        self._3.addWidget(self.wholeCheckBox)

        self.caseCheckBox = QtWidgets.QCheckBox(self.tab)

        self.caseCheckBox.setObjectName("caseCheckBox")

        self._3.addWidget(self.caseCheckBox)

        self.verticalLayout_8.addLayout(self._3)

        self._4 = QtWidgets.QHBoxLayout()

        self._4.setObjectName("_4")

        self.label_3 = QtWidgets.QLabel(self.tab)

        self.label_3.setObjectName("label_3")

        self._4.addWidget(self.label_3)

        self.syntaxComboBox = QtWidgets.QComboBox(self.tab)

        self.syntaxComboBox.setObjectName("syntaxComboBox")

        self.syntaxComboBox.addItem("")

        self.syntaxComboBox.addItem("")

        self._4.addWidget(self.syntaxComboBox)

        self.verticalLayout_8.addLayout(self._4)

        self.line = QtWidgets.QFrame(self.tab)

        self.line.setFrameShape(QtWidgets.QFrame.HLine)

        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.line.setObjectName("line")

        self.verticalLayout_8.addWidget(self.line)

        self.horizontalLayout = QtWidgets.QHBoxLayout()

        self.horizontalLayout.setObjectName("horizontalLayout")

        self.transparencyGroupBox = QtWidgets.QGroupBox(self.tab)

        self.transparencyGroupBox.setCheckable(True)

        self.transparencyGroupBox.setObjectName("transparencyGroupBox")

        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.transparencyGroupBox)

        self.horizontalLayout_11.setObjectName("horizontalLayout_11")

        self.verticalLayout_7 = QtWidgets.QVBoxLayout()

        self.verticalLayout_7.setObjectName("verticalLayout_7")

        self.transparencySlider = QtWidgets.QSlider(self.transparencyGroupBox)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        sizePolicy.setHorizontalStretch(0)

        sizePolicy.setVerticalStretch(0)

        sizePolicy.setHeightForWidth(self.transparencySlider.sizePolicy().hasHeightForWidth())

        self.transparencySlider.setSizePolicy(sizePolicy)

        self.transparencySlider.setMaximum(100)

        self.transparencySlider.setSingleStep(1)

        self.transparencySlider.setProperty("value", 75)

        self.transparencySlider.setOrientation(QtCore.Qt.Horizontal)

        self.transparencySlider.setObjectName("transparencySlider")

        self.verticalLayout_7.addWidget(self.transparencySlider)

        self.onLosingFocusRButton = QtWidgets.QRadioButton(self.transparencyGroupBox)

        self.onLosingFocusRButton.setChecked(True)

        self.onLosingFocusRButton.setObjectName("onLosingFocusRButton")

        self.verticalLayout_7.addWidget(self.onLosingFocusRButton)

        self.alwaysRButton = QtWidgets.QRadioButton(self.transparencyGroupBox)

        self.alwaysRButton.setObjectName("alwaysRButton")

        self.verticalLayout_7.addWidget(self.alwaysRButton)

        self.horizontalLayout_11.addLayout(self.verticalLayout_7)

        self.horizontalLayout.addWidget(self.transparencyGroupBox)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.horizontalLayout.addItem(spacerItem)

        self.verticalLayout_8.addLayout(self.horizontalLayout)

        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.verticalLayout_8.addItem(spacerItem1)

        self.horizontalLayout_12.addLayout(self.verticalLayout_8)

        self.line_2 = QtWidgets.QFrame(self.tab)

        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)

        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.line_2.setObjectName("line_2")

        self.horizontalLayout_12.addWidget(self.line_2)

        self._5 = QtWidgets.QVBoxLayout()

        self._5.setObjectName("_5")

        self.findNextButton = QtWidgets.QPushButton(self.tab)

        self.findNextButton.setObjectName("findNextButton")

        self._5.addWidget(self.findNextButton)

        self.line_5 = QtWidgets.QFrame(self.tab)

        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)

        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.line_5.setObjectName("line_5")

        self._5.addWidget(self.line_5)

        self.findAllInOpenDocsButton = QtWidgets.QPushButton(self.tab)

        self.findAllInOpenDocsButton.setObjectName("findAllInOpenDocsButton")

        self._5.addWidget(self.findAllInOpenDocsButton)

        self.findAllInCurrentDocButton = QtWidgets.QPushButton(self.tab)

        self.findAllInCurrentDocButton.setObjectName("findAllInCurrentDocButton")

        self._5.addWidget(self.findAllInCurrentDocButton)

        self.line_4 = QtWidgets.QFrame(self.tab)

        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)

        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.line_4.setObjectName("line_4")

        self._5.addWidget(self.line_4)

        self.replaceButton = QtWidgets.QPushButton(self.tab)

        self.replaceButton.setObjectName("replaceButton")

        self._5.addWidget(self.replaceButton)

        self.replaceAllButton = QtWidgets.QPushButton(self.tab)

        self.replaceAllButton.setObjectName("replaceAllButton")

        self._5.addWidget(self.replaceAllButton)

        self.inSelectionBox = QtWidgets.QCheckBox(self.tab)

        self.inSelectionBox.setObjectName("inSelectionBox")

        self._5.addWidget(self.inSelectionBox)

        self.replaceAllInOpenDocsButton = QtWidgets.QPushButton(self.tab)

        self.replaceAllInOpenDocsButton.setObjectName("replaceAllInOpenDocsButton")

        self._5.addWidget(self.replaceAllInOpenDocsButton)

        self.line_6 = QtWidgets.QFrame(self.tab)

        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)

        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.line_6.setObjectName("line_6")

        self._5.addWidget(self.line_6)

        spacerItem2 = QtWidgets.QSpacerItem(20, 16, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self._5.addItem(spacerItem2)

        self.closeButton = QtWidgets.QPushButton(self.tab)

        self.closeButton.setObjectName("closeButton")

        self._5.addWidget(self.closeButton)

        self.horizontalLayout_12.addLayout(self._5)

        self.tabWidget.addTab(self.tab, "")

        self.tab_2 = QtWidgets.QWidget()

        self.tab_2.setObjectName("tab_2")

        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.tab_2)

        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)

        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        self.verticalLayout = QtWidgets.QVBoxLayout()

        self.verticalLayout.setObjectName("verticalLayout")

        self.gridLayout = QtWidgets.QGridLayout()

        self.gridLayout.setObjectName("gridLayout")

        self.label_4 = QtWidgets.QLabel(self.tab_2)

        self.label_4.setObjectName("label_4")

        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)

        self.findComboBoxIF = QtWidgets.QComboBox(self.tab_2)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        sizePolicy.setHorizontalStretch(0)

        sizePolicy.setVerticalStretch(0)

        sizePolicy.setHeightForWidth(self.findComboBoxIF.sizePolicy().hasHeightForWidth())

        self.findComboBoxIF.setSizePolicy(sizePolicy)

        self.findComboBoxIF.setMaximumSize(QtCore.QSize(200, 16777215))

        self.findComboBoxIF.setObjectName("findComboBoxIF")

        self.gridLayout.addWidget(self.findComboBoxIF, 0, 1, 1, 1)

        self.label_5 = QtWidgets.QLabel(self.tab_2)

        self.label_5.setObjectName("label_5")

        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1)

        self.replaceComboBoxIF = QtWidgets.QComboBox(self.tab_2)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        sizePolicy.setHorizontalStretch(0)

        sizePolicy.setVerticalStretch(0)

        sizePolicy.setHeightForWidth(self.replaceComboBoxIF.sizePolicy().hasHeightForWidth())

        self.replaceComboBoxIF.setSizePolicy(sizePolicy)

        self.replaceComboBoxIF.setMaximumSize(QtCore.QSize(200, 16777215))

        self.replaceComboBoxIF.setObjectName("replaceComboBoxIF")

        self.gridLayout.addWidget(self.replaceComboBoxIF, 1, 1, 1, 1)

        self.label_6 = QtWidgets.QLabel(self.tab_2)

        self.label_6.setObjectName("label_6")

        self.gridLayout.addWidget(self.label_6, 2, 0, 1, 1)

        self.filtersComboBoxIF = QtWidgets.QComboBox(self.tab_2)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        sizePolicy.setHorizontalStretch(0)

        sizePolicy.setVerticalStretch(0)

        sizePolicy.setHeightForWidth(self.filtersComboBoxIF.sizePolicy().hasHeightForWidth())

        self.filtersComboBoxIF.setSizePolicy(sizePolicy)

        self.filtersComboBoxIF.setMaximumSize(QtCore.QSize(200, 16777215))

        self.filtersComboBoxIF.setObjectName("filtersComboBoxIF")

        self.gridLayout.addWidget(self.filtersComboBoxIF, 2, 1, 1, 1)

        self.label_7 = QtWidgets.QLabel(self.tab_2)

        self.label_7.setObjectName("label_7")

        self.gridLayout.addWidget(self.label_7, 3, 0, 1, 1)

        self.directoryComboBoxIF = QtWidgets.QComboBox(self.tab_2)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        sizePolicy.setHorizontalStretch(0)

        sizePolicy.setVerticalStretch(0)

        sizePolicy.setHeightForWidth(self.directoryComboBoxIF.sizePolicy().hasHeightForWidth())

        self.directoryComboBoxIF.setSizePolicy(sizePolicy)

        self.directoryComboBoxIF.setMaximumSize(QtCore.QSize(200, 16777215))

        self.directoryComboBoxIF.setObjectName("directoryComboBoxIF")

        self.gridLayout.addWidget(self.directoryComboBoxIF, 3, 1, 1, 1)

        self.pickDirectoryButtonIF = QtWidgets.QPushButton(self.tab_2)

        self.pickDirectoryButtonIF.setObjectName("pickDirectoryButtonIF")

        self.gridLayout.addWidget(self.pickDirectoryButtonIF, 3, 2, 1, 1)

        self.verticalLayout.addLayout(self.gridLayout)

        self.gridLayout_2 = QtWidgets.QGridLayout()

        self.gridLayout_2.setObjectName("gridLayout_2")

        self.wholeCheckBoxIF = QtWidgets.QCheckBox(self.tab_2)

        self.wholeCheckBoxIF.setEnabled(True)

        self.wholeCheckBoxIF.setChecked(False)

        self.wholeCheckBoxIF.setObjectName("wholeCheckBoxIF")

        self.gridLayout_2.addWidget(self.wholeCheckBoxIF, 0, 0, 1, 1)

        self.caseCheckBoxIF = QtWidgets.QCheckBox(self.tab_2)

        self.caseCheckBoxIF.setObjectName("caseCheckBoxIF")

        self.gridLayout_2.addWidget(self.caseCheckBoxIF, 0, 1, 1, 1)

        self.verticalLayout.addLayout(self.gridLayout_2)

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()

        self.horizontalLayout_3.setObjectName("horizontalLayout_3")

        self.label_8 = QtWidgets.QLabel(self.tab_2)

        self.label_8.setObjectName("label_8")

        self.horizontalLayout_3.addWidget(self.label_8)

        self.syntaxComboBoxIF = QtWidgets.QComboBox(self.tab_2)

        self.syntaxComboBoxIF.setObjectName("syntaxComboBoxIF")

        self.syntaxComboBoxIF.addItem("")

        self.syntaxComboBoxIF.addItem("")

        self.horizontalLayout_3.addWidget(self.syntaxComboBoxIF)

        self.verticalLayout.addLayout(self.horizontalLayout_3)

        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.verticalLayout.addItem(spacerItem3)

        self.horizontalLayout_4.addLayout(self.verticalLayout)

        self.line_3 = QtWidgets.QFrame(self.tab_2)

        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)

        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.line_3.setObjectName("line_3")

        self.horizontalLayout_4.addWidget(self.line_3)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()

        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.findAllButtonIF = QtWidgets.QPushButton(self.tab_2)

        self.findAllButtonIF.setObjectName("findAllButtonIF")

        self.verticalLayout_2.addWidget(self.findAllButtonIF)

        self.replaceButtonIF = QtWidgets.QPushButton(self.tab_2)

        self.replaceButtonIF.setObjectName("replaceButtonIF")

        self.verticalLayout_2.addWidget(self.replaceButtonIF)

        self.inAllSubFoldersCheckBoxIF = QtWidgets.QCheckBox(self.tab_2)

        self.inAllSubFoldersCheckBoxIF.setEnabled(True)

        self.inAllSubFoldersCheckBoxIF.setChecked(False)

        self.inAllSubFoldersCheckBoxIF.setObjectName("inAllSubFoldersCheckBoxIF")

        self.verticalLayout_2.addWidget(self.inAllSubFoldersCheckBoxIF)

        spacerItem4 = QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(spacerItem4)

        self.closeButtonIF = QtWidgets.QPushButton(self.tab_2)

        self.closeButtonIF.setObjectName("closeButtonIF")

        self.verticalLayout_2.addWidget(self.closeButtonIF)

        self.horizontalLayout_4.addLayout(self.verticalLayout_2)

        self.tabWidget.addTab(self.tab_2, "")

        self.tab_3 = QtWidgets.QWidget()

        self.tab_3.setObjectName("tab_3")

        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_3)

        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)

        self.gridLayout_3.setObjectName("gridLayout_3")

        self.groupBox = QtWidgets.QGroupBox(self.tab_3)

        self.groupBox.setObjectName("groupBox")

        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox)

        self.verticalLayout_4.setObjectName("verticalLayout_4")

        self.verticalLayout_3 = QtWidgets.QVBoxLayout()

        self.verticalLayout_3.setObjectName("verticalLayout_3")

        self.findCPB = QtWidgets.QPushButton(self.groupBox)

        self.findCPB.setObjectName("findCPB")

        self.verticalLayout_3.addWidget(self.findCPB)

        self.replaceCPB = QtWidgets.QPushButton(self.groupBox)

        self.replaceCPB.setObjectName("replaceCPB")

        self.verticalLayout_3.addWidget(self.replaceCPB)

        self.filtersCPB = QtWidgets.QPushButton(self.groupBox)

        self.filtersCPB.setObjectName("filtersCPB")

        self.verticalLayout_3.addWidget(self.filtersCPB)

        self.directoryCPB = QtWidgets.QPushButton(self.groupBox)

        self.directoryCPB.setObjectName("directoryCPB")

        self.verticalLayout_3.addWidget(self.directoryCPB)

        self.verticalLayout_4.addLayout(self.verticalLayout_3)

        self.gridLayout_3.addWidget(self.groupBox, 0, 0, 1, 1)

        spacerItem5 = QtWidgets.QSpacerItem(199, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout_3.addItem(spacerItem5, 0, 1, 1, 1)

        spacerItem6 = QtWidgets.QSpacerItem(20, 90, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.gridLayout_3.addItem(spacerItem6, 1, 0, 1, 1)

        self.tabWidget.addTab(self.tab_3, "")

        self.horizontalLayout_7.addWidget(self.tabWidget)

        self.label_2.setBuddy(self.replaceComboBox)

        self.label.setBuddy(self.findComboBox)

        self.label_3.setBuddy(self.syntaxComboBox)

        self.label_4.setBuddy(self.findComboBox)

        self.label_5.setBuddy(self.replaceComboBox)

        self.label_8.setBuddy(self.syntaxComboBox)



        self.retranslateUi(FindInFiles)

        self.tabWidget.setCurrentIndex(0)

        self.closeButton.clicked.connect(FindInFiles.close)

        self.closeButtonIF.clicked.connect(FindInFiles.reject)

        QtCore.QMetaObject.connectSlotsByName(FindInFiles)

        FindInFiles.setTabOrder(self.findComboBox, self.replaceComboBox)

        FindInFiles.setTabOrder(self.replaceComboBox, self.wholeCheckBox)

        FindInFiles.setTabOrder(self.wholeCheckBox, self.syntaxComboBox)

        FindInFiles.setTabOrder(self.syntaxComboBox, self.findNextButton)

        FindInFiles.setTabOrder(self.findNextButton, self.replaceButton)

        FindInFiles.setTabOrder(self.replaceButton, self.replaceAllButton)

        FindInFiles.setTabOrder(self.replaceAllButton, self.closeButton)

        FindInFiles.setTabOrder(self.closeButton, self.findComboBoxIF)

        FindInFiles.setTabOrder(self.findComboBoxIF, self.replaceComboBoxIF)

        FindInFiles.setTabOrder(self.replaceComboBoxIF, self.filtersComboBoxIF)

        FindInFiles.setTabOrder(self.filtersComboBoxIF, self.caseCheckBox)

        FindInFiles.setTabOrder(self.caseCheckBox, self.directoryComboBoxIF)

        FindInFiles.setTabOrder(self.directoryComboBoxIF, self.pickDirectoryButtonIF)

        FindInFiles.setTabOrder(self.pickDirectoryButtonIF, self.wholeCheckBoxIF)

        FindInFiles.setTabOrder(self.wholeCheckBoxIF, self.caseCheckBoxIF)

        FindInFiles.setTabOrder(self.caseCheckBoxIF, self.syntaxComboBoxIF)

        FindInFiles.setTabOrder(self.syntaxComboBoxIF, self.findAllButtonIF)

        FindInFiles.setTabOrder(self.findAllButtonIF, self.replaceButtonIF)

        FindInFiles.setTabOrder(self.replaceButtonIF, self.inAllSubFoldersCheckBoxIF)

        FindInFiles.setTabOrder(self.inAllSubFoldersCheckBoxIF, self.closeButtonIF)



    def retranslateUi(self, FindInFiles):

        _translate = QtCore.QCoreApplication.translate

        FindInFiles.setWindowTitle(_translate("FindInFiles", "Find and Replace"))

        self.label_2.setText(_translate("FindInFiles", "Replace w&ith:"))

        self.label.setText(_translate("FindInFiles", "Find &what:"))

        self.wholeCheckBox.setText(_translate("FindInFiles", "Wh&ole words"))

        self.caseCheckBox.setText(_translate("FindInFiles", "&Case sensitive"))

        self.label_3.setText(_translate("FindInFiles", "&Syntax:"))

        self.syntaxComboBox.setItemText(0, _translate("FindInFiles", "Literal text"))

        self.syntaxComboBox.setItemText(1, _translate("FindInFiles", "Regular expression"))

        self.transparencyGroupBox.setTitle(_translate("FindInFiles", "Transparency"))

        self.onLosingFocusRButton.setText(_translate("FindInFiles", "On Losing Focus"))

        self.alwaysRButton.setText(_translate("FindInFiles", "Always"))

        self.findNextButton.setText(_translate("FindInFiles", "Find Next"))

        self.findAllInOpenDocsButton.setText(_translate("FindInFiles", "Find All in Open Docs"))

        self.findAllInCurrentDocButton.setText(_translate("FindInFiles", "Find All in Current Doc"))

        self.replaceButton.setText(_translate("FindInFiles", "&Replace"))

        self.replaceAllButton.setText(_translate("FindInFiles", "Replace &All"))

        self.inSelectionBox.setText(_translate("FindInFiles", "In Selection"))

        self.replaceAllInOpenDocsButton.setText(_translate("FindInFiles", "Replace All in Open Docs"))

        self.closeButton.setText(_translate("FindInFiles", "Close"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("FindInFiles", "Find and Replace"))

        self.label_4.setText(_translate("FindInFiles", "Find &what:"))

        self.label_5.setText(_translate("FindInFiles", "Replace w&ith:"))

        self.label_6.setText(_translate("FindInFiles", "Filters:"))

        self.label_7.setText(_translate("FindInFiles", "Directory:"))

        self.pickDirectoryButtonIF.setText(_translate("FindInFiles", "..."))

        self.wholeCheckBoxIF.setText(_translate("FindInFiles", "Wh&ole words"))

        self.caseCheckBoxIF.setText(_translate("FindInFiles", "&Case sensitive"))

        self.label_8.setText(_translate("FindInFiles", "&Syntax:"))

        self.syntaxComboBoxIF.setItemText(0, _translate("FindInFiles", "Literal text"))

        self.syntaxComboBoxIF.setItemText(1, _translate("FindInFiles", "Regular expression"))

        self.findAllButtonIF.setText(_translate("FindInFiles", "Find All"))

        self.replaceButtonIF.setText(_translate("FindInFiles", "&Replace in Files"))

        self.inAllSubFoldersCheckBoxIF.setText(_translate("FindInFiles", "In all sub-folders"))

        self.closeButtonIF.setText(_translate("FindInFiles", "Close"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("FindInFiles", "Find and Replace in Files"))

        self.groupBox.setTitle(_translate("FindInFiles", "Options"))

        self.findCPB.setText(_translate("FindInFiles", "Clear Find History"))

        self.replaceCPB.setText(_translate("FindInFiles", "Clear Replace History"))

        self.filtersCPB.setText(_translate("FindInFiles", "Clear Filters History"))

        self.directoryCPB.setText(_translate("FindInFiles", "Clear Directory History"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("FindInFiles", "Clear Data"))



