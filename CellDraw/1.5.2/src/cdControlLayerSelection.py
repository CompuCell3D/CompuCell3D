#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ======================================================================
# 2011 - Mitja: select "layer mode" with radio-buttons inside a box:
#               (a QGroupBox-based control)
# ======================================================================
# note: this class emits one signal:
#
#         signalLayersSelectionModeHasChanged = QtCore.pyqtSignal(int)
#
class CDControlLayerSelection(QtGui.QWidget):

    # ------------------------------------------------------------

    signalLayersSelectionModeHasChanged = QtCore.pyqtSignal(int)

    # ------------------------------------------------------------
    def __init__(self,parent=None):
        QtGui.QWidget.__init__(self, parent)

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
        self.theLayerMode = CDConstants.SceneModeMoveItem

        # ----------------------------------------------------------------
        #
        # QWidget setup (1) - windowing GUI setup for Control Panel:
        #

        self.setWindowTitle("Layer Selection Window Title")
        # QVBoxLayout layout lines up widgets vertically:
        self.layerSelectionMainLayout = QtGui.QVBoxLayout()
        self.layerSelectionMainLayout.setMargin(2)
        self.layerSelectionMainLayout.setSpacing(4)
        self.layerSelectionMainLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
#         self.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
#         self.setAutoFillBackground(True)
        self.setLayout(self.layerSelectionMainLayout)

        # Prepare the font for the radio buttons' caption text:
        lFont = QtGui.QFont()

        # Setting font sizes for Qt widget does NOT work correctly across platforms,
        #   for example the following setPointSize() shows a smaller-than-standard
        #   font on Mac OS X, but it shows a larger-than-standard font on Linux.
        #   Therefore setPointSize() can't be used directly like this:
        # lFont.setPointSize(11)
        lFont.setWeight(QtGui.QFont.Light)


        # ----------------------------------------------------------------
        #
        # QWidget setup (2) - prepare three radio buttons, one for each
        #   distinct "layer mode" and the SceneModeMoveItem is set as the default.
        #
        CDConstants.printOut( "___ - DEBUG ----- CDControlLayerSelection: populateControlPanel() 1", CDConstants.DebugTODO )

        self.layerSelectionGroupBox = QtGui.QGroupBox("Edit Mode")
        # self.layerSelectionGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
        # self.layerSelectionGroupBox.setAutoFillBackground(True)
        self.layerSelectionGroupBox.setLayout(QtGui.QHBoxLayout())
        self.layerSelectionGroupBox.layout().setMargin(2)
        self.layerSelectionGroupBox.layout().setSpacing(4)
        self.layerSelectionGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # create buttons for the QGroupBox:
       
        # regular Scene layer pointer button, for "move" mode in Scene layer:
        self.pointerButton = QtGui.QToolButton()
        self.pointerButton.setCheckable(True)
        self.pointerButton.setChecked(True)
        self.pointerButton.setIcon(QtGui.QIcon(':/icons/pointer.png'))
        self.pointerButton.setIconSize(QtCore.QSize(24, 24))
        self.pointerButton.setToolTip("Scene Layer - Select Tool")
        self.pointerButton.setStatusTip("Scene Layer Select Tool: select a region in the Cell Scene")

        # button to switch to "resize" mode in the Scene Layer:
        self.resizeButton = QtGui.QToolButton()
        self.resizeButton.setCheckable(True)
        self.resizeButton.setChecked(False)
        self.resizeButton.setIcon(QtGui.QIcon(':/icons/resizepointer.png'))
        self.resizeButton.setIconSize(QtCore.QSize(24, 24))
        self.resizeButton.setToolTip("Scene Layer - Resize Tool")
        self.resizeButton.setStatusTip("Scene Layer Resize Tool: resize a region in the Cell Scene")


# 2012 - Mitja: the following three buttons (imageLayerButton, imageSequenceButton, editClusterButton)
#   have been hidden from the control panel GUI, since their selection is performed by clicking on
#   on the tab related to their functionality --- find out where the tabs are, and that the above two
#   buttons (pointerButton, resizeButton) are also correctly related to their related tab.!!!

        # a new button to show the image layer:
        self.imageLayerButton = QtGui.QToolButton()
        self.imageLayerButton.setCheckable(True)
        self.imageLayerButton.setChecked(False)
        self.imageLayerButton.setIcon(QtGui.QIcon(':/icons/imageLayer.png'))
        self.imageLayerButton.setIconSize(QtCore.QSize(24, 24))
        self.imageLayerButton.setToolTip("Image Layer")
        self.imageLayerButton.setStatusTip("Image Layer: pick regions, draw on top of image")
        self.imageLayerButton.hide()

        # a new button to show the image sequence layer:
        self.imageSequenceButton = QtGui.QToolButton()
        self.imageSequenceButton.setCheckable(True)
        self.imageSequenceButton.setChecked(False)
        self.imageSequenceButton.setIcon(QtGui.QIcon(':/icons/imageSequence.png'))
        self.imageSequenceButton.setIconSize(QtCore.QSize(24, 24))
        self.imageSequenceButton.setToolTip("Image Sequence Layer")
        self.imageSequenceButton.setStatusTip("Image Sequence Layer: access a stack of images in a sequence")
        self.imageSequenceButton.hide()

        # a new button to show the image sequence layer:
        self.editClusterButton = QtGui.QToolButton()
        self.editClusterButton.setCheckable(True)
        self.editClusterButton.setChecked(False)
        self.editClusterButton.setIcon(QtGui.QIcon(':/icons/editCluster.png'))
        self.editClusterButton.setIconSize(QtCore.QSize(24, 24))
        self.editClusterButton.setToolTip("Cell Cluster Editor")
        self.editClusterButton.setStatusTip("Cell Cluster Editor: prepare a blueprint for clusters of cells")
        self.editClusterButton.hide()

        # add all buttons to the QGroupBox:
        self.layerSelectionGroupBox.layout().addWidget(self.pointerButton)
        self.layerSelectionGroupBox.layout().addWidget(self.resizeButton)       
        self.layerSelectionGroupBox.layout().addWidget(self.imageLayerButton)
        self.layerSelectionGroupBox.layout().addWidget(self.imageSequenceButton)
        self.layerSelectionGroupBox.layout().addWidget(self.editClusterButton)
        # finally add the QGroupBox  to the main layout in the widget:
        self.layerSelectionMainLayout.addWidget(self.layerSelectionGroupBox)


        # ----------------------------------------------------------------
        #
        # QWidget setup (3) - "Layer Selection" QButtonGroup,
        #    a *logical* container to make buttons mutually exclusive:

        self.theButtonGroupForLayerSelection = QtGui.QButtonGroup()
        self.theButtonGroupForLayerSelection.addButton(self.pointerButton, CDConstants.SceneModeMoveItem)
        self.theButtonGroupForLayerSelection.addButton(self.resizeButton, CDConstants.SceneModeResizeItem)
        self.theButtonGroupForLayerSelection.addButton(self.imageLayerButton, CDConstants.SceneModeImageLayer)
        self.theButtonGroupForLayerSelection.addButton(self.imageSequenceButton, CDConstants.SceneModeImageSequence)
        self.theButtonGroupForLayerSelection.addButton(self.editClusterButton, CDConstants.SceneModeEditCluster)
        # self.theButtonGroupForLayerSelection.addButton(linePointerButton, CDConstants.SceneModeInsertLine)

        # call handleLayerButtonGroupClicked() every time a button is clicked in the "theButtonGroupForLayerSelection"
        self.theButtonGroupForLayerSelection.buttonClicked[int].connect( \
            self.handleLayerButtonGroupClicked)



        # setWindowOpacity seems to work only if it's set after setting WindowFlags and attributes:
        self.setWindowOpacity(0.95)



    # ------------------------------------------------------------
    # return the ID of the only checked button in the QButtonGroup:
    # ------------------------------------------------------------
    def getCheckedButtonId(self):
        return self.theButtonGroupForLayerSelection.checkedId()


    # ------------------------------------------------------------
    # set a checked button in the QButtonGroup:
    # ------------------------------------------------------------
    def setCheckedButton(self, pId, pChecked=True):
        self.theButtonGroupForLayerSelection.button(pId).setChecked(pChecked)


    # ------------------------------------------------------------
    # programmatically click on a button in the QButtonGroup:
    # ------------------------------------------------------------
    def clickOnButton(self, pId):
        self.theButtonGroupForLayerSelection.button(pId).click()


    # ------------------------------------------------------------
    # set the icon of the Image Layer selection button
    # ------------------------------------------------------------
    def setImageLayerButtonIcon(self, pIcon):
        self.imageLayerButton.setIcon(QtGui.QIcon( pIcon ))



    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "buttonClicked" events
    #    (AKA signals) arriving from theButtonGroupForLayerSelection:
    # ------------------------------------------------------------
    def handleLayerButtonGroupClicked(self, pChecked):
        if self.pointerButton.isChecked():
            lLayerMode = CDConstants.SceneModeMoveItem
        elif self.resizeButton.isChecked():
            lLayerMode = CDConstants.SceneModeResizeItem
        elif self.imageLayerButton.isChecked():
            lLayerMode = CDConstants.SceneModeImageLayer
        elif self.imageSequenceButton.isChecked():
            lLayerMode = CDConstants.SceneModeImageSequence
        elif self.editClusterButton.isChecked():
            lLayerMode = CDConstants.SceneModeEditCluster
        if lLayerMode != self.theLayerMode:
            self.theLayerMode = lLayerMode
            CDConstants.printOut("CDControlLayerSelection - handleLayerButtonGroupClicked(), the layer mode is = " +str(self.theLayerMode), CDConstants.DebugVerbose)
            # propagate the signal upstream, for example to parent objects:
            self.signalLayersSelectionModeHasChanged.emit(self.theLayerMode)



# end class CDControlLayerSelection(QtGui.QWidget)
# ======================================================================
