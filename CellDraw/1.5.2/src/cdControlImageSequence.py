#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

import sys    # sys is necessary to inquire about "sys.version_info"

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ======================================================================
# a QWidget-based control
# ======================================================================


# ------------------------------------------------------------
# 2011 - Mitja:  "image sequence" controls inside a box:
# ------------------------------------------------------------
# note: this class emits these signals:
#
#         signalSelectedImageInSequenceHasChanged = QtCore.pyqtSignal(int)
#
#         signalImageSequenceProcessingModeHasChanged = QtCore.pyqtSignal(int)
#
#         signalSetCurrentTypeColor = QtCore.pyqtSignal(int)
#
#         signalForPIFFTableToggle = QtCore.pyqtSignal(str)
#
class CDControlImageSequence(QtGui.QWidget):

    # ------------------------------------------------------------

    signalSelectedImageInSequenceHasChanged = QtCore.pyqtSignal(int)

    signalImageSequenceProcessingModeHasChanged = QtCore.pyqtSignal(int)

    signalSetCurrentTypeColor = QtCore.pyqtSignal(QtCore.QVariant)

    # the signal used to toggle visibility of the Table of Types:
    signalForPIFFTableToggle = QtCore.pyqtSignal(str)

    # ------------------------------------------------------------

    def __init__(self,parent=None):
        QtGui.QWidget.__init__(self, parent)
        CDConstants.printOut("___ - DEBUG ----- CDControlImageSequence: __init__() ", CDConstants.DebugTODO )


        # the class global keeping track of the selected image within the sequence:
        #    0 = minimum = the first image in the sequence stack
        self.imageCurrentIndex = 0

        # the class global keeping track of the mode for generating PIFF from displayed imported image sequence:
        #    0 = Use Discretized Images to B/W = CDConstants.ImageSequenceUseDiscretizedToBWMode
        #    1 = Region 2D Edge = CDConstants.ImageSequenceUse2DEdges
        #    2 = Region 3D Contours = CDConstants.ImageSequenceUse3DContours
        #    3 = Region 3D Volume = CDConstants.ImageSequenceUse3DVolume
        #    4 = Region Cell Seeds = CDConstants.ImageSequenceUseAreaSeeds
        self.theImageSequenceProcessingMode = (1 << CDConstants.ImageSequenceUseAreaSeeds)

        # bin() does not exist in Python 2.5:
        if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
            CDConstants.printOut( "___ - DEBUG ----- CDControlImageSequence: __init__() bin(self.theImageSequenceProcessingMode) == "+str(bin(self.theImageSequenceProcessingMode)) , CDConstants.DebugExcessive )
        else:
            CDConstants.printOut( "___ - DEBUG ----- CDControlImageSequence: __init__() self.theImageSequenceProcessingMode == "+str(self.theImageSequenceProcessingMode) , CDConstants.DebugExcessive )

        # typical usage :
        # | is used to set a certain bit to 1
        # & is used to test or clear a certaint bit
        # 
        # Set a bit (where n is the bit number, and 0 is the least significant bit):
        # a |= (1 << n)
        # 
        # Clear a bit:
        # b &= ~(1 << n)
        # 
        # Toggle a bit:
        # c ^= (1 << n)
        # 
        # Test a bit:
        # e = d & (1 << n)


        #
        # QWidget setup (1) - windowing GUI setup for Control Panel:
        #

        self.setWindowTitle("Image Sequence Window Title")
        # QVBoxLayout layout lines up widgets vertically:
        self.imageSequenceControlsMainLayout = QtGui.QVBoxLayout()
        self.imageSequenceControlsMainLayout.setContentsMargins(0,0,0,0)
        self.imageSequenceControlsMainLayout.setSpacing(4)
        self.imageSequenceControlsMainLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.setLayout(self.imageSequenceControlsMainLayout)

        lFont = QtGui.QFont()
        lFont.setWeight(QtGui.QFont.Light)


        # ----------------------------------------------------------------

        self.controlChoosenImageGroupBox = QtGui.QGroupBox("Selected Image in Sequence")
        self.controlChoosenImageGroupBox.setLayout(QtGui.QHBoxLayout())
        self.controlChoosenImageGroupBox.layout().setContentsMargins(0,0,0,0)
        self.controlChoosenImageGroupBox.layout().setSpacing(4)
        self.controlChoosenImageGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # 2011 - Mitja: a QSlider to control the selected image in the sequence:
        self.imageSelectionSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.imageSelectionSlider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.imageSelectionSlider.setTickPosition(QtGui.QSlider.NoTicks)
        self.imageSelectionSlider.setSingleStep(1)
        self.imageSelectionSlider.setMinimum(0)
        self.imageSelectionSlider.setMaximum(99)
        self.imageSelectionSlider.setValue(0)
        self.imageSelectionSlider.setStatusTip("Image Selection: choose the specific image within the sequence to be displayed.")
        self.imageSelectionSlider.setToolTip("Image Selection")
        self.imageSelectionSlider.valueChanged.connect(self.handleImageSelectionSliderChanged)

        self.imageSelectionLabel = QtGui.QLabel()
        self.imageSelectionLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lIntValue = int(self.imageSelectionSlider.value())
        lStrValue = str(lIntValue)
        self.imageSelectionLabel.setText(lStrValue)
        self.imageSelectionLabel.setMinimumSize( QtCore.QSize(24, 24) )
        self.imageSelectionLabel.setContentsMargins(0,0,0,0)
        self.imageSelectionLabel.setFont(lFont)

        self.controlChoosenImageGroupBox.layout().addWidget(self.imageSelectionSlider)
        self.controlChoosenImageGroupBox.layout().addWidget(self.imageSelectionLabel)


        # ----------------------------------------------------------------

        self.areaOrEdgeSelectionGroupBox = QtGui.QGroupBox("Cell Seeds|2D Edges|3D Contours|Volume| B/W")
        self.areaOrEdgeSelectionGroupBox.setLayout(QtGui.QHBoxLayout())
        self.areaOrEdgeSelectionGroupBox.layout().setContentsMargins(0,0,0,0)
        self.areaOrEdgeSelectionGroupBox.layout().setSpacing(4)
        self.areaOrEdgeSelectionGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # create buttons for the QGroupBox:
       
        # region seeds button, for generating cell seeds for the inside area:
        self.regionSeedsButton = QtGui.QToolButton()
        self.regionSeedsButton.setCheckable(True)
        self.regionSeedsButton.setChecked(True)
        self.regionSeedsButton.setIcon(QtGui.QIcon(':/icons/imageSequenceShowSeeds.png'))
        self.regionSeedsButton.setIconSize(QtCore.QSize(32, 32))
        self.regionSeedsButton.setToolTip("Generate Cell Seeds in Volume from Image Sequence")
        self.regionSeedsButton.setStatusTip("Image Sequence Region Area: generate pixel seeds for cells within the 3D volume defined from the imported sequence")

        # button for generating a solid 3D volume single-ID:
        self.regionVolumeButton = QtGui.QToolButton()
        self.regionVolumeButton.setCheckable(True)
        self.regionVolumeButton.setChecked(False)
        self.regionVolumeButton.setIcon(QtGui.QIcon(':/icons/imageSequenceShowVolume.png'))
        self.regionVolumeButton.setIconSize(QtCore.QSize(32, 32))
        self.regionVolumeButton.setToolTip("Generate 3D Volume from Image Sequence")
        self.regionVolumeButton.setStatusTip("Image Sequence Volume: generate a solid 3D volume, single-cell-ID, defined from the imported sequence")

        # button for generating a full-3D contour of the volume based on each image region's edge area:
        self.regionContoursButton = QtGui.QToolButton()
        self.regionContoursButton.setCheckable(True)
        self.regionContoursButton.setChecked(False)
        self.regionContoursButton.setIcon(QtGui.QIcon(':/icons/imageSequenceShowContours.png'))
        self.regionContoursButton.setIconSize(QtCore.QSize(32, 32))
        self.regionContoursButton.setToolTip("Generate 3D Contour from Image Sequence")
        self.regionContoursButton.setStatusTip("Image Sequence 3D Contour: generate a 3D contour of the volume defined from the imported sequence")

        # button for generating a 2D contour for each image region's edge area:
        self.regionEdgeButton = QtGui.QToolButton()
        self.regionEdgeButton.setCheckable(True)
        self.regionEdgeButton.setChecked(False)
        self.regionEdgeButton.setIcon(QtGui.QIcon(':/icons/imageSequenceShowEdge.png'))
        self.regionEdgeButton.setIconSize(QtCore.QSize(32, 32))
        self.regionEdgeButton.setToolTip("Generate 2D Edges from Image Sequence")
        self.regionEdgeButton.setStatusTip("Image Sequence 2D Edges: generate a 2D contour for each image region's edge area in the imported sequence")

        # checkbox to toggle black/white discretization of image sequence
        self.discretizeToBWModeButton = QtGui.QToolButton()
        self.discretizeToBWModeButton.setCheckable(True)
        self.discretizeToBWModeButton.setChecked(False)
        self.discretizeToBWModeButton.setIcon(QtGui.QIcon(':/icons/imageSequenceBW.png'))
        self.discretizeToBWModeButton.setIconSize(QtCore.QSize(32, 32))
        self.discretizeToBWModeButton.setToolTip("B/W : treat the image sequence as black/white images.")
        self.discretizeToBWModeButton.setStatusTip("B/W: Treat the image sequence as black/white images: all non-black pixels are considered as white.")



        # add all buttons to the QGroupBox:
        self.areaOrEdgeSelectionGroupBox.layout().addWidget(self.regionSeedsButton)
        self.areaOrEdgeSelectionGroupBox.layout().addWidget(self.regionVolumeButton)
        self.areaOrEdgeSelectionGroupBox.layout().addWidget(self.regionContoursButton)       
        self.areaOrEdgeSelectionGroupBox.layout().addWidget(self.regionEdgeButton)       
        self.areaOrEdgeSelectionGroupBox.layout().addSpacing(20)
        self.areaOrEdgeSelectionGroupBox.layout().addWidget(self.discretizeToBWModeButton)





        # ----------------------------------------------------------------
        #    a *logical* container of the above buttons:

        self.theButtonGroupForAreaOrEdgeSelection = QtGui.QButtonGroup()
        self.theButtonGroupForAreaOrEdgeSelection.addButton(self.regionSeedsButton, CDConstants.ImageSequenceUseAreaSeeds)
        self.theButtonGroupForAreaOrEdgeSelection.addButton(self.regionVolumeButton, CDConstants.ImageSequenceUse3DVolume)
        self.theButtonGroupForAreaOrEdgeSelection.addButton(self.regionContoursButton, CDConstants.ImageSequenceUse3DContours)
        self.theButtonGroupForAreaOrEdgeSelection.addButton(self.regionEdgeButton, CDConstants.ImageSequenceUse2DEdges)
        self.theButtonGroupForAreaOrEdgeSelection.addButton(self.discretizeToBWModeButton, CDConstants.ImageSequenceUseDiscretizedToBWMode)
        # make sure that the buttons are *not* mutually exclusive:
        self.theButtonGroupForAreaOrEdgeSelection.setExclusive(False)

        # call handleAreaOrEdgeButtonGroupClicked() every time a button is clicked in the "theButtonGroupForAreaOrEdgeSelection"
        self.theButtonGroupForAreaOrEdgeSelection.buttonClicked[int].connect( \
            self.handleAreaOrEdgeButtonGroupClicked)










        # ----------------------------------------------------------------
        #
        #   prepare two separate buttons:
        #    one for fill color with a pop-up menu,
        #    and another one to toggle the table of types
        #

        self.typesGroupBox = QtGui.QGroupBox("PIFF Types")
        self.typesGroupBox.setLayout(QtGui.QHBoxLayout())
        self.typesGroupBox.layout().setMargin(2)
        self.typesGroupBox.layout().setSpacing(4)
        self.typesGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # the fillColorToolButton is a pop-up menu button,
        #    Menu defaults are set here and for consistency they *must* coincide
        #    with the defaults set in the DiagramScene class globals.
        #
        self.fillColorToolButton = QtGui.QToolButton()
        self.fillColorToolButton.setText("Color")

        self.fillColorToolButton.setIcon( \
            self.createFloodFillToolButtonIcon( \
                ':/icons/floodfill.png', \
                QtGui.QColor(QtCore.Qt.green)    )   )
        self.fillColorToolButton.setIconSize(QtCore.QSize(24, 24))

        self.fillColorToolButton.setStatusTip("Set the sequence's types")
        self.fillColorToolButton.setToolTip("Set the sequence's types")

        self.fillColorToolButton.setPopupMode(QtGui.QToolButton.MenuButtonPopup)

        # attach a popup menu to the button, with event handler handleFillColorChanged()
        self.fillColorToolButton.setMenu(  \
            self.createColorMenu(self.handleFillColorChanged, \
            QtCore.Qt.green)  )

        self.fillColorToolButton.menu().setStatusTip("Set the sequence's types (Menu)")
        self.fillColorToolButton.menu().setToolTip("Set the sequence's types (Menu)")

        self.fillAction = self.fillColorToolButton.menu().defaultAction()

        # provide a "slot" function to the button, the event handler is handleFillButtonClicked()
        self.fillColorToolButton.clicked.connect(self.handleFillButtonClicked)



        # ------------------------------------------------------------
        # 2011 - the fillColorToolButton is a button acting as toggle,
        #   with the pifTableAction used to show/hide the Table of Types window:
        # Note: PyQt 4.8.6 seems to have problems with assigning the proper key shortcuts
        #   using mnemonics such as  shortcut=QtGui.QKeySequence.AddTab, so we have to 
        #   set the shortcut explicitly to "Ctrl+key" ...
        self.pifTableAction = QtGui.QAction( \
                QtGui.QIcon(':/icons/regiontable.png'), "Table of Types", self, \
                shortcut="Ctrl+T", statusTip="Toggle (show/hide) the Table of Types window", \
                triggered=self.handlePIFTableButton)

        # add an action to the QGroupBox:
        lToolButton = QtGui.QToolButton(self)
        lToolButton.setDefaultAction(self.pifTableAction)
        lToolButton.setCheckable(self.pifTableAction.isCheckable())
        lToolButton.setChecked(self.pifTableAction.isChecked())
        lToolButton.setIcon(self.pifTableAction.icon())
        lToolButton.setIconSize(QtCore.QSize(24, 24))
        lToolButton.setToolTip(self.pifTableAction.toolTip())
        lToolButton.setStatusTip(self.pifTableAction.statusTip())

        # add all buttons to the QGroupBox:
        
        # TODO: don't add the self.fillColorToolButton until we have the code to handle it !!!!!
        # self.typesGroupBox.layout().addWidget(self.fillColorToolButton)

        self.typesGroupBox.layout().addWidget(lToolButton)


        # ----------------------------------------------------------------
        # place all QGroupBox items in the main controls layout:
        #
        self.imageSequenceControlsMainLayout.addWidget(self.controlChoosenImageGroupBox)
        self.imageSequenceControlsMainLayout.addWidget(self.areaOrEdgeSelectionGroupBox)
        self.imageSequenceControlsMainLayout.addWidget(self.typesGroupBox)








    # ------------------------------------------------------------
    # 2011 - Mitja: creating a "Color" menu produces a QMenu item,
    #    which allows the creation of a color-picking menu:
    #
    # ------------------------------------------------------------
    def createColorMenu(self, pSlotFunction, pDefaultColor):
        lColorMenu = QtGui.QMenu(self)
        for lColor, lName in zip(CDConstants.TypesColors, CDConstants.TypesColorNames):
            # CDConstants.printOut( " "+str( "lColor =", lColor, "lName =", lName )+" ", CDConstants.DebugTODO )
            lAction = QtGui.QAction(self.createColorIcon(lColor),
                      QtCore.QString(lName), self, triggered=pSlotFunction)
            # set the action's data to be the color:
            lAction.setData(QtGui.QColor(lColor))
            lColorMenu.addAction(lAction)
            if lColor == pDefaultColor:
                lColorMenu.setDefaultAction(lAction)
        return lColorMenu

    # ------------------------------------------------------------
    def createColorIcon(self, color):
        pixmap = QtGui.QPixmap(20, 20)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtCore.Qt.NoPen)
        painter.fillRect(QtCore.QRect(0, 0, 20, 20), QtGui.QBrush(QtGui.QColor(color)))
        painter.end()
        return QtGui.QIcon(pixmap)

    # ------------------------------------------------------------
    # 2011 - Mitja: add code for creating better looking flood fill button:
    # ------------------------------------------------------------
    def createFloodFillToolButtonIcon(self, pImageFile, pColor):
        lPixmap = QtGui.QPixmap(80, 80)
        lPixmap.fill(QtCore.Qt.transparent)
        lPainter = QtGui.QPainter(lPixmap)
        lImage = QtGui.QPixmap(pImageFile)
        lTarget = QtCore.QRect(0, 0, 60, 60)
        lSource = QtCore.QRect(0, 0, 44, 44)
        lPainter.fillRect(QtCore.QRect(0, 60, 80, 80), pColor)
        lPainter.drawPixmap(lTarget, lImage, lSource)
        lPainter.end()
        return QtGui.QIcon(lPixmap)







    # ------------------------------------------------------------
    def setMaxImageIndex(self, pValueInt):
        self.imageSelectionSlider.setMaximum(pValueInt)
        self.imageSelectionSlider.update()
        CDConstants.printOut("CDControlImageSequence - setMaxImageIndex( " +str(pValueInt)+" ) done.", CDConstants.DebugVerbose)











    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "triggered" events
    #    (AKA signals) arriving from the fillColorToolButton menu:
    # ------------------------------------------------------------
    def handleFillColorChanged(self):

        self.fillAction = self.sender()
        self.chosenMenuColor = self.fillAction.data()
        # CDConstants.printOut( " "+str( "self.chosenMenuColor is now", self.chosenMenuColor, "not", QtGui.QColor(self.chosenMenuColor) )+" ", CDConstants.DebugTODO )
       
        self.fillColorToolButton.setIcon( \
            self.createFloodFillToolButtonIcon(':/icons/floodfill.png', \
            QtGui.QColor(self.chosenMenuColor)  )  )
        self.fillColorToolButton.setIconSize(QtCore.QSize(24, 24))

        # propagate the signal upstream, for example to parent objects:
        self.signalSetCurrentTypeColor.emit( QtGui.QColor(self.chosenMenuColor) )



    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "clicked" events
    #    (AKA signals) arriving from the fillColorToolButton button:
    # ------------------------------------------------------------
    def handleFillButtonClicked(self):
        # there is no change in chosen color when the button is clicked, so
        #    propagate the signal upstream, for example to parent objects:
        self.signalSetCurrentTypeColor.emit( QtGui.QColor(self.chosenMenuColor) )



    # ------------------------------------------------------------
    # 2010 Mitja - slot/method handling events/signals from pifTableAction:
    # ------------------------------------------------------------
    def handlePIFTableButton(self):
        self.signalForPIFFTableToggle.emit("Toggle")










    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "valueChanged" events
    #    (AKA signals) arriving from imageSelectionSlider:
    # ------------------------------------------------------------
    def handleImageSelectionSliderChanged(self, pValueInt):
        lSelectedImage = pValueInt
        if (lSelectedImage != self.imageCurrentIndex) :
            self.imageCurrentIndex = pValueInt
            lStrValue = str(self.imageCurrentIndex)
            self.imageSelectionLabel.setText(lStrValue)
            # propagate the signal upstream, for example to parent objects:
            self.signalSelectedImageInSequenceHasChanged.emit(self.imageCurrentIndex)



    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "buttonClicked" events
    #    (AKA signals) arriving from theButtonGroupForAreaOrEdgeSelection:
    # ------------------------------------------------------------
    def handleAreaOrEdgeButtonGroupClicked(self, pChecked):
        lImageSequenceProcessingMode = 0

        if self.regionSeedsButton.isChecked():
            lImageSequenceProcessingMode |= (1 << CDConstants.ImageSequenceUseAreaSeeds)
        if self.regionEdgeButton.isChecked():
            lImageSequenceProcessingMode |= (1 << CDConstants.ImageSequenceUse2DEdges)
        if self.regionContoursButton.isChecked():
            lImageSequenceProcessingMode |= (1 << CDConstants.ImageSequenceUse3DContours)
        if self.regionVolumeButton.isChecked():
            lImageSequenceProcessingMode |= (1 << CDConstants.ImageSequenceUse3DVolume)
        if self.discretizeToBWModeButton.isChecked():
            lImageSequenceProcessingMode |= (1 << CDConstants.ImageSequenceUseDiscretizedToBWMode)

#         TODO add fourth mode here, in the sequence class, and in the rasterizer class:
#         
#         then TODO change PIFF numbering format printout specifically only for image sequence rasterization:

        if lImageSequenceProcessingMode != self.theImageSequenceProcessingMode:
            self.theImageSequenceProcessingMode = lImageSequenceProcessingMode

            # bin() does not exist in Python 2.5:
            if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
                CDConstants.printOut( "CDControlImageSequence - handleAreaOrEdgeButtonGroupClicked(), str(type(self.theImageSequenceProcessingMode))==["+str(type(self.theImageSequenceProcessingMode))+"], str(type(self.theImageSequenceProcessingMode).__name__)==["+str(type(self.theImageSequenceProcessingMode).__name__)+"], str(self.theImageSequenceProcessingMode)==["+str(self.theImageSequenceProcessingMode)+"], str(bin(int(self.theImageSequenceProcessingMode)))==["+str(bin(int(self.theImageSequenceProcessingMode)))+"]" , CDConstants.DebugTODO )
                CDConstants.printOut("CDControlImageSequence - handleAreaOrEdgeButtonGroupClicked(), theImageSequenceProcessingMode is = " +str(bin(self.theImageSequenceProcessingMode)), CDConstants.DebugVerbose)
            else:
                CDConstants.printOut( "CDControlImageSequence - handleAreaOrEdgeButtonGroupClicked(), str(type(self.theImageSequenceProcessingMode))==["+str(type(self.theImageSequenceProcessingMode))+"], str(type(self.theImageSequenceProcessingMode).__name__)==["+str(type(self.theImageSequenceProcessingMode).__name__)+"], str(self.theImageSequenceProcessingMode)==["+str(self.theImageSequenceProcessingMode)+"], str(int(self.theImageSequenceProcessingMode))==["+str(int(self.theImageSequenceProcessingMode))+"]" , CDConstants.DebugTODO )
                CDConstants.printOut("CDControlImageSequence - handleAreaOrEdgeButtonGroupClicked(), theImageSequenceProcessingMode is = " +str(self.theImageSequenceProcessingMode), CDConstants.DebugVerbose)

            # propagate the signal upstream, for example to parent objects:
            self.signalImageSequenceProcessingModeHasChanged.emit(self.theImageSequenceProcessingMode)



# end class CDControlImageSequence(QtGui.QWidget)
# ======================================================================
