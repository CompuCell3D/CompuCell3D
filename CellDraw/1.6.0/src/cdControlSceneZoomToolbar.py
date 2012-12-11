#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ======================================================================
# 2011 - Mitja: set scene zoom with a combo box (pop-up menu) :
#               (inside a box, a QGroupBox-based control)
# ======================================================================
# note: this class emits one signal:
#
#         signalZoomFactorHasChanged = QtCore.pyqtSignal(str)
#
class CDControlSceneZoomToolbar(QtGui.QToolBar):

    # ------------------------------------------------------------

    signalZoomFactorHasChanged = QtCore.pyqtSignal(str)

    # ------------------------------------------------------------
    def __init__(self,pString=None,pParent=None):

        if (pString==None):
            lString=QtCore.QString("CDControlSceneZoomToolbar QToolBar")
        else:
            lString=QtCore.QString(pString)
        QtGui.QToolBar.__init__(self, lString, pParent)

        # the class global keeping track of the current scene zoom value:
        self.__theZoomFactor = "100%"

        # ----------------------------------------------------------------
        #
        # QWidget setup (1) - windowing GUI setup for Scene Zoom controls:
        #




        # ----------------------------------------------------------------
        #
        # QWidget setup (2) - "Zoom" QGroupBox:

        self.__theZoomGroupBox = QtGui.QGroupBox("Zoom Scene")
        #         self.__theZoomGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
        #         self.__theZoomGroupBox.setAutoFillBackground(True)
        # set the position of the QGroupBox's label:
        self.__theZoomGroupBox.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
        self.__theZoomGroupBox.setFlat(False)
        self.__theZoomGroupBox.setLayout(QtGui.QHBoxLayout())
        self.__theZoomGroupBox.layout().setMargin(0)
        self.__theZoomGroupBox.layout().setSpacing(0)
        self.__theZoomGroupBox.layout().setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        # a "combo box" pop-up menu to select the Zoom Scene factor:
        self.__sceneZoomComboBox = QtGui.QComboBox()
        self.__sceneZoomComboBox.addItems(["50%", "75%", "100%", "125%", "150%", "200%", "250%", "300%", "400%", "500%", "1000%", "2000%", "4000%"])
        self.__sceneZoomComboBox.setCurrentIndex(2)
        self.__sceneZoomComboBox.clearFocus()
        self.__sceneZoomComboBox.setStatusTip("Zoom the Cell Scene view")
        self.__sceneZoomComboBox.setToolTip("Zoom Cell Scene")
        self.__sceneZoomComboBox.setFrame(False)
       
        # call __handleSceneZoomChanged() when __sceneZoomComboBox changes index:
        self.__sceneZoomComboBox.currentIndexChanged[str].connect(self.__handleSceneZoomChanged)

        # add the combo box to the QGroupBox:
        self.__theZoomGroupBox.layout().addWidget(self.__sceneZoomComboBox)

        # finally add the QGroupBox  to the QToolBar:
#         self.addWidget(QtGui.QLabel("["))
        self.addWidget(self.__theZoomGroupBox)
#         self.addWidget(QtGui.QLabel("]"))

        self.show()

        CDConstants.printOut( "----- CDControlSceneZoomToolbar.__init__(pString=="+str(pString)+", pParent=="+str(pParent)+") done. -----", CDConstants.DebugExcessive )


        # setWindowOpacity seems to work only if it's set after setting WindowFlags and attributes:
#         self.setWindowOpacity(0.95)




    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "currentIndexChanged" events
    #    (AKA signals) arriving from __sceneZoomComboBox:
    # ------------------------------------------------------------
    def __handleSceneZoomChanged(self, pValueString):

        CDConstants.printOut( "the requested Scene zoom is = "+str(pValueString), CDConstants.DebugTODO )
        lZoomFactor = pValueString
        if lZoomFactor != self.__theZoomFactor:
            self.__theZoomFactor = lZoomFactor
            CDConstants.printOut( "the new Scene zoom will be = "+str(self.__theZoomFactor), CDConstants.DebugTODO )
            # propagate the signal upstream, for example to parent objects:
            self.signalZoomFactorHasChanged.emit(self.__theZoomFactor)


    # ------------------------------------------------------------
    # register the callback handler function for the
    #   "signalZoomFactorHasChanged()" signal:
    # ------------------------------------------------------------
    def registerHandlerForToolbarChanges(self, pHandler):
        self.signalZoomFactorHasChanged.connect( pHandler )



# end class CDControlSceneZoomToolbar(QtGui.QToolBar)
# ======================================================================
