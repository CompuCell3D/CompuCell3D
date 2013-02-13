#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ======================================================================
# a QWidget-based view, to display the cell scene dimensions:
# ======================================================================
class CDViewSceneDimensions(QtGui.QWidget):

    # ------------------------------------------------------------
    def __init__(self,pParent=None):

        QtGui.QWidget.__init__(self, pParent)

        # ----------------------------------------------------------------
        #
        # QWidget setup (1) - windowing GUI setup for view:
        #


        lFont = QtGui.QFont()
        lFont.setWeight(QtGui.QFont.Light)


        # ----------------------------------------------------------------
        #
        # QWidget setup (2) - add a QGroupBox for showing Scene dimensions:
        #
        self.__sceneDimensionsGroupBox = QtGui.QGroupBox("Scene Dimensions")
#         self.__sceneDimensionsGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,0,222)))
#         self.__sceneDimensionsGroupBox.setAutoFillBackground(True)
        self.__sceneDimensionsGroupBox.setLayout(QtGui.QHBoxLayout())
        self.__sceneDimensionsGroupBox.layout().setMargin(0)
        self.__sceneDimensionsGroupBox.layout().setSpacing(0)
        self.__sceneDimensionsGroupBox.layout().setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        # if we set lFont.setWeight(QtGui.QFont.Light) to the QGroupBox, it actually becomes bigger!
        # self.__sceneDimensionsGroupBox.setFont(lFont)


        # 2010 - Mitja: add a widget displaying the scene dimensions at all times:
        # the scene dimension widget will have a title label:
        self.__sceneWidthLabel = QtGui.QLabel()
        self.__sceneWidthLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.__sceneWidthLabel.setText("w")
        self.__sceneWidthLabel.setFont(lFont)
        self.__sceneWidthLabel.setMargin(2)
        sceneTimesSignLabel = QtGui.QLabel()
        sceneTimesSignLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneTimesSignLabel.setText( u"\u00D7" ) # <-- the multiplication sign as unicode
        sceneTimesSignLabel.setFont(lFont)
        sceneTimesSignLabel.setMargin(2)
        self.__sceneHeightLabel = QtGui.QLabel()
        self.__sceneHeightLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.__sceneHeightLabel.setText("h")
        self.__sceneHeightLabel.setFont(lFont)
        self.__sceneHeightLabel.setMargin(2)
        sceneTimesSign2Label = QtGui.QLabel()
        sceneTimesSign2Label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneTimesSign2Label.setText( u"\u00D7" ) # <-- the multiplication sign as unicode
        sceneTimesSign2Label.setFont(lFont)
        sceneTimesSign2Label.setMargin(2)
        self.__sceneDepthLabel = QtGui.QLabel()
        self.__sceneDepthLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.__sceneDepthLabel.setText("d")
        self.__sceneDepthLabel.setFont(lFont)
        self.__sceneDepthLabel.setMargin(2)
        self.__sceneUnitsLabel = QtGui.QLabel()
        self.__sceneUnitsLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.__sceneUnitsLabel.setText("(units)")
        self.__sceneUnitsLabel.setFont(lFont)
        self.__sceneUnitsLabel.setMargin(2)


        self.sceneDimensionsWidget = QtGui.QWidget()
#         self.sceneDimensionsWidget.setPalette(QtGui.QPalette(QtGui.QColor(222,222,0)))
#         self.sceneDimensionsWidget.setAutoFillBackground(True)
        self.sceneDimensionsWidget.setLayout(QtGui.QHBoxLayout())
        self.sceneDimensionsWidget.layout().setMargin(0)
        self.sceneDimensionsWidget.layout().setSpacing(0)
        self.sceneDimensionsWidget.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.sceneDimensionsWidget.setFont(lFont)
        self.sceneDimensionsWidget.layout().addWidget(self.__sceneWidthLabel)
        self.sceneDimensionsWidget.layout().addWidget(sceneTimesSignLabel)
        self.sceneDimensionsWidget.layout().addWidget(self.__sceneHeightLabel)
        self.sceneDimensionsWidget.layout().addWidget(sceneTimesSign2Label)
        self.sceneDimensionsWidget.layout().addWidget(self.__sceneDepthLabel)
        self.sceneDimensionsWidget.layout().addWidget(self.__sceneUnitsLabel)


        self.__sceneDimensionsGroupBox.layout().addWidget(self.sceneDimensionsWidget)


#         self.setPalette(QtGui.QPalette(QtGui.QColor(0,222,222)))
#         self.setAutoFillBackground(True)
        self.setFont(lFont)
        self.setLayout(QtGui.QHBoxLayout())
        self.layout().setMargin(0)
        self.layout().setSpacing(0)
        self.layout().setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        __statusBarString = QtCore.QString("Scene Dimensions: width "+u"\u00D7"+" height "+u"\u00D7"+" depth ")
        self.setStatusTip(__statusBarString)
        self.setToolTip("Scene Dimensions")

        self.layout().addWidget(self.__sceneDimensionsGroupBox)


    # end of    def __init__(self,pParent=None)
    # ------------------------------------------------------------


    # ------------------------------------------------------------------
    # functions for externally setting the control panel's label values:

    # ------------------------------------------------------------------
    def setSceneWidthLabel(self, pSceneWidthLabel):
        self.__sceneWidthLabel.setText(pSceneWidthLabel)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setSceneWidthLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setSceneHeightLabel(self, pSceneHeightLabel):
        self.__sceneHeightLabel.setText(pSceneHeightLabel)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setSceneHeightLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setSceneDepthLabel(self, pSceneDepthLabel):
        self.__sceneDepthLabel.setText(pSceneDepthLabel)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setSceneHeightLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setSceneUnitsLabel(self, pSceneUnitsLabel):
        self.__sceneUnitsLabel.setText(pSceneUnitsLabel)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setSceneUnitsLabel(): done" )+" ", CDConstants.DebugTODO )



    # ------------------------------------------------------------
    # 2011 - Mitja: moving to a MVC design,
    #   this connect signal handler should go to the Controller object!
    # ------------------------------------------------------------
    def handlerForSceneResized(self, pDict):
        lDict = dict(pDict)
        CDConstants.printOut("  ", CDConstants.DebugTODO )
        CDConstants.printOut("    TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO ", CDConstants.DebugTODO )
        CDConstants.printOut(str( lDict ) , CDConstants.DebugTODO )
        CDConstants.printOut("    TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO ", CDConstants.DebugTODO )
        CDConstants.printOut("  ", CDConstants.DebugTODO )
        self.setSceneWidthLabel(lDict[0])
        self.setSceneHeightLabel(lDict[1])
        self.setSceneDepthLabel(lDict[2])
        self.setSceneUnitsLabel(lDict[3])

# end class CDViewSceneDimensions(QtGui.QWidget)
# ======================================================================
