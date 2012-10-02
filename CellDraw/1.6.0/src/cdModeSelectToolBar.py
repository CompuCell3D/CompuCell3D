#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants

# the following import is about a resource file generated thus:
#     "pyrcc4 cdDiagramScene.qrc -o cdDiagramScene_rc.py"
# which requires the file cdDiagramScene.qrc to correctly point to the files in ":/images"
# only that way will the icons etc. be available after the import:
import cdDiagramScene_rc


# ======================================================================
# 2011 - Mitja: select "layer mode" with radio-buttons inside a box:
#               (a QGroupBox-based control)
# ======================================================================
# note: this class emits one signal:
#
#         signalModeSelectToolbarChanged = QtCore.pyqtSignal(int)
#
class CDModeSelectToolBar(QtGui.QToolBar):

    # ------------------------------------------------------------

    signalModeSelectToolbarChanged = QtCore.pyqtSignal(int)

    # ------------------------------------------------------------
    def __init__(self,pString=None,pParent=None):

        if (pString==None):
            lString=QtCore.QString("CDModeSelectToolBar QToolBar")
        else:
            lString=QtCore.QString(pString)
        QtGui.QToolBar.__init__(self, lString, pParent)

        # the class global keeping track of the selected picking mode:
        #    CDConstants.SceneModeInsertItem,
        #    CDConstants.SceneModeInsertLine,
        #    CDConstants.SceneModeInsertText,
        #    CDConstants.SceneModeMoveItem,      <---
        #    CDConstants.SceneModeInsertPixmap,
        #    CDConstants.SceneModeResizeItem,    <---
        #    CDConstants.SceneModeImageLayer     <---
        #    CDConstants.SceneModeImageSequence     <---
        #    CDConstants.SceneModeEditCluster     <---
        #    = range(8)

        self.selectedSceneMode = CDConstants.SceneModeMoveItem
        self.sceneModeActionDict = dict()

        # ----------------------------------------------------------------
        #
        # QToolBar setup (1) - windowing GUI setup for Layer Selection controls:
        #

        # self.setWindowTitle("TooBar - Layer Selection - Window Title")

        # set all icons in this QToolBar to be of the same size:
        self.setIconSize(QtCore.QSize(24, 24))


# QVBoxLayout layout lines up widgets vertically:
#         self.layerSelectionMainLayout = QtGui.QVBoxLayout()
#         self.layerSelectionMainLayout.setMargin(2)
#         self.layerSelectionMainLayout.setSpacing(4)
#         self.layerSelectionMainLayout.setAlignment( \
#             QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
# #         self.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
# #         self.setAutoFillBackground(True)
#         print "00"
#         self.setLayout(self.layerSelectionMainLayout)
#         print "01"

        # Prepare the font for the radio buttons' caption text:
# TODOTODOTODO        lFont = QtGui.QFont()

        # Setting font sizes for Qt widget does NOT work correctly across platforms,
        #   for example the following setPointSize() shows a smaller-than-standard
        #   font on Mac OS X, but it shows a larger-than-standard font on Linux.
        #   Therefore setPointSize() can't be used directly like this:
        # lFont.setPointSize(11)
# TODOTODOTODO        lFont.setWeight(QtGui.QFont.Light)


        # ----------------------------------------------------------------
        #
        # QToolBar setup (2) - prepare radio buttons, one for each "layer mode":
        #
        CDConstants.printOut( "___ - DEBUG ----- CDModeSelectToolBar: __init__() 1", CDConstants.DebugExcessive )

        # ----------------------------------------------------------------
        #
        # QToolBar setup (3) - "Layer Selection" QActionGroup,
        #    a *logical* container to make buttons mutually exclusive:

        self.theActionGroupForLayerSelection = QtGui.QActionGroup(self)
        self.theActionGroupForLayerSelection.setExclusive(True)
        # call handleLayerActionGroupTriggered() every time a button is clicked in the "theActionGroupForLayerSelection":
        self.theActionGroupForLayerSelection.triggered[QtGui.QAction].connect(self.handleLayerActionGroupTriggered)

        # create actions for the QGroupBox:
       
        # regular Scene layer pointer button, for "move" mode in Scene layer:
        self.pointerAction = QtGui.QAction(self.theActionGroupForLayerSelection)
        self.pointerAction.setCheckable(True)
        self.pointerAction.setChecked(True)
        self.pointerAction.setIcon(QtGui.QIcon(':/icons/pointer.png'))
#        self.pointerAction.setIconSize(QtCore.QSize(24, 24))
        self.pointerAction.setToolTip("Scene Layer - Select Tool")
        self.pointerAction.setStatusTip("Scene Layer Select Tool: select a region in the Cell Scene")
        self.sceneModeActionDict[CDConstants.SceneModeMoveItem] = self.pointerAction

        # button to switch to "resize" mode in the Scene Layer:
        self.resizeAction = QtGui.QAction(self.theActionGroupForLayerSelection)
        self.resizeAction.setCheckable(True)
        self.resizeAction.setChecked(False)
        self.resizeAction.setIcon(QtGui.QIcon(':/icons/resizepointer.png'))
#        self.resizeAction.setIconSize(QtCore.QSize(24, 24))
        self.resizeAction.setToolTip("Scene Layer - Resize Tool")
        self.resizeAction.setStatusTip("Scene Layer Resize Tool: resize a region in the Cell Scene")
        self.sceneModeActionDict[CDConstants.SceneModeResizeItem] = self.resizeAction


# TODO TODO:
#
# 2012 - Mitja: the following three buttons (imageLayerAction, imageSequenceAction, editClusterAction)
#   have been hidden from the control panel GUI, since their selection is performed by clicking on
#   on the tab related to their functionality --- find out where the tabs are, and that the above two
#   buttons (pointerAction, resizeAction) are also correctly related to their related tab.!!!

        # a new button to show the image layer:
        self.imageLayerAction = QtGui.QAction(self.theActionGroupForLayerSelection)
        self.imageLayerAction.setCheckable(True)
        self.imageLayerAction.setChecked(False)
        self.imageLayerAction.setIcon(QtGui.QIcon(':/icons/imageLayer.png'))
#        self.imageLayerAction.setIconSize(QtCore.QSize(24, 24))
        self.imageLayerAction.setToolTip("Image Layer")
        self.imageLayerAction.setStatusTip("Image Layer: pick regions, draw on top of image")
        self.sceneModeActionDict[CDConstants.SceneModeImageLayer] = self.imageLayerAction

        # a new button to show the image sequence layer:
        self.imageSequenceAction = QtGui.QAction(self.theActionGroupForLayerSelection)
        self.imageSequenceAction.setCheckable(True)
        self.imageSequenceAction.setChecked(False)
        self.imageSequenceAction.setIcon(QtGui.QIcon(':/icons/imageSequence.png'))
#        self.imageSequenceAction.setIconSize(QtCore.QSize(24, 24))
        self.imageSequenceAction.setToolTip("Image Sequence Layer")
        self.imageSequenceAction.setStatusTip("Image Sequence Layer: access a stack of images in a sequence")
#         self.imageSequenceAction.hide()
        self.sceneModeActionDict[CDConstants.SceneModeImageSequence] = self.imageSequenceAction

        # a new button to show the image sequence layer:
#         self.editClusterAction = QtGui.QAction(self.theActionGroupForLayerSelection)
#         self.editClusterAction.setCheckable(True)
#         self.editClusterAction.setChecked(False)
#         self.editClusterAction.setIcon(QtGui.QIcon(':/icons/editCluster.png'))
# #        self.editClusterAction.setIconSize(QtCore.QSize(24, 24))
#         self.editClusterAction.setToolTip("Cell Cluster Editor")
#         self.editClusterAction.setStatusTip("Cell Cluster Editor: prepare a blueprint for clusters of cells")
#         self.editClusterAction.hide()

        # add all buttons to the QToolBar:
        self.addAction(self.pointerAction)
        self.addAction(self.resizeAction)       
        self.addAction(self.imageLayerAction)
        self.addAction(self.imageSequenceAction)

        self.show()



    # ------------------------------------------------------------
    # return the ID of the only checked button in the QButtonGroup:
    # ------------------------------------------------------------
    def getSelectedSceneMode(self):
        return self.selectedSceneMode


    # ------------------------------------------------------------
    # set a checked button in the QButtonGroup:
    # ------------------------------------------------------------
    def setSelectedSceneMode(self, pId):
        self.selectedSceneMode = pId
        self.sceneModeActionDict[self.selectedSceneMode].setChecked(True)

# 
#     # ------------------------------------------------------------
#     # programmatically click on a button in the QButtonGroup:
#     # ------------------------------------------------------------
#     def clickOnButton(self, pId):
#         self.theActionGroupForLayerSelection.button(pId).click()
# 

    # ------------------------------------------------------------
    # set the icon of the Image Layer selection button
    # ------------------------------------------------------------
    def setImageLayerButtonIcon(self, pIcon):
        self.imageLayerAction.setIcon(QtGui.QIcon( pIcon ))



    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "triggered" events
    #    (AKA signals) arriving from theActionGroupForLayerSelection:
    # ------------------------------------------------------------
    def handleLayerActionGroupTriggered(self, pChecked):
        if self.pointerAction.isChecked():
            lLayerMode = CDConstants.SceneModeMoveItem
        elif self.resizeAction.isChecked():
            lLayerMode = CDConstants.SceneModeResizeItem
        elif self.imageLayerAction.isChecked():
            lLayerMode = CDConstants.SceneModeImageLayer
        elif self.imageSequenceAction.isChecked():
            lLayerMode = CDConstants.SceneModeImageSequence
#         elif self.editClusterAction.isChecked():
#             lLayerMode = CDConstants.SceneModeEditCluster
        if lLayerMode != self.selectedSceneMode:
            self.selectedSceneMode = lLayerMode
            CDConstants.printOut("CDModeSelectToolBar - handleLayerActionGroupTriggered(), the layer mode is = " +str(self.selectedSceneMode), CDConstants.DebugVerbose)
            # propagate the signal upstream, for example to parent objects:
            self.signalModeSelectToolbarChanged.emit(self.selectedSceneMode)


    # ------------------------------------------------------------
    # register the callback handler function for the
    #   "signalModeSelectToolbarChanged()" signal:
    # ------------------------------------------------------------
    def registerSignalHandlerForModeSelectToolbarChanges(self, pHandler):
        self.signalModeSelectToolbarChanged.connect( pHandler )

# end class CDModeSelectToolBar(QtGui.QToolBar)
# ======================================================================





# ------------------------------------------------------------
# just for testing:
# ------------------------------------------------------------
if __name__ == '__main__':
    CDConstants.printOut( "___ - DEBUG ----- CDModeSelectToolBar: __main__() 1", CDConstants.DebugTODO )
    import sys

    app = QtGui.QApplication(sys.argv)

    CDConstants.printOut( "___ - DEBUG ----- CDModeSelectToolBar: __main__() 2", CDConstants.DebugTODO )

    testQMainWindow = QtGui.QMainWindow()
    testQMainWindow.setGeometry(100, 100, 900, 500)

    CDConstants.printOut( "___ - DEBUG ----- CDModeSelectToolBar: __main__() 3", CDConstants.DebugTODO )

    print "testQMainWindow = ", testQMainWindow
    lTestToolBarObject = CDModeSelectToolBar("lTestToolBarObject Title",testQMainWindow)
    print "lTestToolBarObject = ", lTestToolBarObject
   
    testQMainWindow.addToolBar(QtCore.Qt.TopToolBarArea, lTestToolBarObject)
    print "NOW testQMainWindow.addToolBar(QtCore.Qt.TopToolBarArea,lTestToolBarObject) ..."
    print testQMainWindow.addToolBar(QtCore.Qt.TopToolBarArea,lTestToolBarObject)
    testQMainWindow.setUnifiedTitleAndToolBarOnMac(False)

    CDConstants.printOut( "___ - DEBUG ----- CDModeSelectToolBar: __main__() 4", CDConstants.DebugTODO )

    # 2010 - Mitja: QMainWindow.raise_() must be called after QMainWindow.show()
    #     otherwise the PyQt/Qt-based GUI won't receive foreground focus.
    #     It's a workaround for a well-known bug caused by PyQt/Qt on Mac OS X
    #     as shown here:
    #       http://www.riverbankcomputing.com/pipermail/pyqt/2009-September/024509.html
    testQMainWindow.raise_()
    testQMainWindow.show()

    CDConstants.printOut( "___ - DEBUG ----- CDModeSelectToolBar: __main__() 5", CDConstants.DebugTODO )

    sys.exit(app.exec_())

    CDConstants.printOut( "___ - DEBUG ----- CDModeSelectToolBar: __main__() 6", CDConstants.DebugTODO )

# Local Variables:
# coding: US-ASCII
# End:
