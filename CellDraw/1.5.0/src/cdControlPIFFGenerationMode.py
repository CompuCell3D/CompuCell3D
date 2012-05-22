#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ======================================================================
# 2011 - Mitja: select "PIFF generation mode" with radio-buttons inside a box:
#               (a QGroupBox-based control)
# ======================================================================
# note: this class emits one signal to be used outside:
#
#         signalPIFFGenerationModeHasChanged = QtCore.pyqtSignal(int,int)
#
class CDControlPIFFGenerationMode(QtGui.QWidget):

    # ------------------------------------------------------------

    signalPIFFGenerationModeHasChanged = QtCore.pyqtSignal(int, int)

    # ------------------------------------------------------------
    def __init__(self,parent=None):
        QtGui.QWidget.__init__(self, parent)

        # the class global keeping track of the selected PIFF generation mode:
        #    CDConstants.PIFFSaveWithFixedRaster,
        #    CDConstants.PIFFSaveWithOneRasterPerRegion,
        #    CDConstants.PIFFSaveWithPotts = range(3)
        self.thePIFFGenerationMode = CDConstants.PIFFSaveWithOneRasterPerRegion
       
        # the class global keeping track of the fixed-raster width:
        self.thePIFFFixedRasterWidth = 10

        # ----------------------------------------------------------------
        #
        # QWidget setup (1) - windowing GUI setup:
        #

        # QVBoxLayout layout lines up widgets horizontally:
        self.layerSelectionMainLayout = QtGui.QHBoxLayout()
        self.layerSelectionMainLayout.setContentsMargins(0,0,0,0)
        self.layerSelectionMainLayout.setSpacing(0)
        self.layerSelectionMainLayout.setAlignment( \
            QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
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
        #   distinct "PIFF Generation Mode" and the PIFFSaveWithOneRasterPerRegion is set as the default.
        #
        print "___ - DEBUG ----- CDControlPIFFGenerationMode: __init__() 1"

        self.layerSelectionGroupBox = QtGui.QGroupBox("PIFF")
        self.layerSelectionGroupBox.setLayout(QtGui.QVBoxLayout())
        self.layerSelectionGroupBox.layout().setContentsMargins(2,2,2,2)
        self.layerSelectionGroupBox.layout().setSpacing(4)
        self.layerSelectionGroupBox.layout().setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)

        # create buttons for the QGroupBox:
       
        # fixed raster PIFF Generation button:
        self.fixedRasterButton = QtGui.QToolButton()
        self.fixedRasterButton.setCheckable(True)
        self.fixedRasterButton.setChecked(False)
        self.fixedRasterButton.setIcon(QtGui.QIcon(':/icons/PIFFFixedRaster.png'))
        self.fixedRasterButton.setIconSize(QtCore.QSize(24, 24))
        self.fixedRasterButton.setText("Fixed")
        lStringForToolTip = "PIFF Generation - Fixed Raster\n sets all PIFF cell regions\n to be made of same-sized square cells,\n as from the fixed raster size."
        self.fixedRasterButton.setToolTip(lStringForToolTip)
        self.fixedRasterButton.setStatusTip("PIFF Generation: save all cells in all regions as squares on a fixed raster")

        # region-rasters-based ("variable") PIFF Generation button:
        self.oneRasterPerRegionButton = QtGui.QToolButton()
        self.oneRasterPerRegionButton.setCheckable(True)
        self.oneRasterPerRegionButton.setChecked(True)
        self.oneRasterPerRegionButton.setIcon(QtGui.QIcon(':/icons/PIFFOneRasterPerRegion.png'))
        self.oneRasterPerRegionButton.setIconSize(QtCore.QSize(24, 24))
        self.oneRasterPerRegionButton.setText("Region")
        lStringForToolTip = "PIFF Generation - Region Rasters\n sets each region's cells\n to be made of region-specific rectangular cells,\n as from the Table of Types"
        self.oneRasterPerRegionButton.setToolTip(lStringForToolTip)
        self.oneRasterPerRegionButton.setStatusTip("PIFF Generation: save cells using separate rasters for each scene region")

        # Potts CC3D-based PIFF Generation button:
        self.usePottsForPIFFButton = QtGui.QToolButton()
        self.usePottsForPIFFButton.setCheckable(True)
        self.usePottsForPIFFButton.setChecked(False)
        self.usePottsForPIFFButton.setIcon(QtGui.QIcon(':/icons/PIFFPottsRaster.png'))
        self.usePottsForPIFFButton.setIconSize(QtCore.QSize(24, 24))
        self.usePottsForPIFFButton.setText("Potts")
        lStringForToolTip = "PIFF Generation - Potts Model\n will prepare cells to be computed\n using the Potts algorithm from CC3D."
        self.usePottsForPIFFButton.setToolTip(lStringForToolTip)
        self.usePottsForPIFFButton.setStatusTip("PIFF Generation: save cells using the Potts Model to achieve specific cell volumes for each cell type")


        # add all buttons  to the QGroupBox:
        self.layerSelectionGroupBox.layout().addWidget(self.fixedRasterButton)
        self.layerSelectionGroupBox.layout().addWidget(self.oneRasterPerRegionButton)       
        self.layerSelectionGroupBox.layout().addWidget(self.usePottsForPIFFButton)
        # finally add the QGroupBox  to the main layout in the widget:
        self.layerSelectionMainLayout.addWidget(self.layerSelectionGroupBox)


        # ----------------------------------------------------------------
        #
        # QWidget setup (3) - create a "PIFF Generation Mode" QButtonGroup,
        #    a *logical* container to make buttons mutually exclusive,
        #    and assign to each button an output value (from CDConstants) :
        #
        self.theButtonGroupForPIFFGenerationSelection = QtGui.QButtonGroup()
        self.theButtonGroupForPIFFGenerationSelection.addButton(self.fixedRasterButton, CDConstants.PIFFSaveWithFixedRaster)
        self.theButtonGroupForPIFFGenerationSelection.addButton(self.oneRasterPerRegionButton, CDConstants.PIFFSaveWithOneRasterPerRegion)
        self.theButtonGroupForPIFFGenerationSelection.addButton(self.usePottsForPIFFButton, CDConstants.PIFFSaveWithPotts)

        # call handleLayerButtonGroupClicked() every time a button is clicked in the "theButtonGroupForPIFFGenerationSelection"
        self.theButtonGroupForPIFFGenerationSelection.buttonClicked[int].connect( \
            self.handleLayerButtonGroupClicked)

        # setWindowOpacity seems to work only if it's set after setting WindowFlags and attributes:
        self.setWindowOpacity(0.95)



    # ------------------------------------------------------------
    # return the ID of the only checked button in the QButtonGroup:
    # ------------------------------------------------------------
    def getCheckedButtonId(self):
        return self.theButtonGroupForPIFFGenerationSelection.checkedId()


    # ------------------------------------------------------------
    # set a checked button in the QButtonGroup:
    # ------------------------------------------------------------
    def setCheckedButton(self, pId, pChecked=True):
        self.theButtonGroupForPIFFGenerationSelection.button(pId).setChecked(pChecked)


    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "buttonClicked" events
    #    (AKA signals) arriving from theButtonGroupForPIFFGenerationSelection:
    # ------------------------------------------------------------
    def handleLayerButtonGroupClicked(self, pChecked):

        if self.fixedRasterButton.isChecked():
            lPIFFGenerationMode = CDConstants.PIFFSaveWithFixedRaster

            # fixed raster PIFF Generation button:
            rasterWidthSpinBox = QtGui.QSpinBox()
            rasterWidthSpinBox.setMinimum(4)
            rasterWidthSpinBox.setValue(self.thePIFFFixedRasterWidth)
            lStringForToolTip = "PIFF Generation - Fixed Raster Width\n sets all PIFF cell regions\n to be made of same-sized square cells."
            rasterWidthSpinBox.setToolTip(lStringForToolTip)
            rasterWidthSpinBox.setStatusTip("PIFF Generation: save all cells in all regions as squares on a fixed raster")
            # call setFixedRasterWidth() every time the rasterWidthSpinBox changes its value:
            rasterWidthSpinBox.valueChanged[int].connect(self.setFixedRasterWidth)

            lPopUpWindow = QtGui.QDialog(self)
            lPopUpWindow.setLayout(QtGui.QHBoxLayout())
            lPopUpWindow.layout().addWidget(QtGui.QLabel("Fixed Raster Width:"))
            lPopUpWindow.layout().addWidget(rasterWidthSpinBox)
            lPopUpWindow.setWindowFlags(QtCore.Qt.Tool | QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.CustomizeWindowHint)
            lPopUpWindow.setAttribute(QtCore.Qt.WindowModal  | \
                QtCore.Qt.WA_MacNoClickThrough | \
                QtCore.Qt.WA_MacVariableSize  )
            lPopUpWindow.setMinimumSize(256, 64)
            lPopUpWindow.show()

        elif self.oneRasterPerRegionButton.isChecked():
            lPIFFGenerationMode = CDConstants.PIFFSaveWithOneRasterPerRegion

        elif self.usePottsForPIFFButton.isChecked():
            lPIFFGenerationMode = CDConstants.PIFFSaveWithPotts

        if lPIFFGenerationMode != self.thePIFFGenerationMode:
            self.thePIFFGenerationMode = lPIFFGenerationMode
            print "the PIFF generation mode is now =", self.thePIFFGenerationMode
            # propagate the signal upstream, for example to parent objects:
            self.signalPIFFGenerationModeHasChanged.emit( \
                self.thePIFFGenerationMode, self.thePIFFFixedRasterWidth)


    # ------------------------------------------------------------------
    # 2011 Mitja - this is a slot method to handle "content change" events
    #    (AKA signals) arriving from rasterWidthSpinBox
    # ------------------------------------------------------------------
    def setFixedRasterWidth(self, pTheNewValue):
        if pTheNewValue != self.thePIFFFixedRasterWidth:
            self.thePIFFFixedRasterWidth = pTheNewValue
            print "the PIFF fixed raster width is now =", self.thePIFFFixedRasterWidth
            # propagate the signal upstream, for example to parent objects:
            self.signalPIFFGenerationModeHasChanged.emit( \
                self.thePIFFGenerationMode, self.thePIFFFixedRasterWidth)

# end class CDControlPIFFGenerationMode(QtGui.QWidget)
# ======================================================================
