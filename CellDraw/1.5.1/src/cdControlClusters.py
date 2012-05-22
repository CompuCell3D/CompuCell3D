#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants

from cdPixelEditor import PixelEditor


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
#         signalSetCurrentTypeColor = QtCore.pyqtSignal(int)
#
#         signalForPIFFTableToggle = QtCore.pyqtSignal(str)
#
class CDControlClusters(QtGui.QWidget):

    # ------------------------------------------------------------

    signalSetCurrentTypeColor = QtCore.pyqtSignal(QtCore.QVariant)

    signalSelectedImageInSequenceHasChanged = QtCore.pyqtSignal(int)

    # the signal used to toggle visibility of the Table of Types:
    signalForPIFFTableToggle = QtCore.pyqtSignal(str)

    # ------------------------------------------------------------

    def __init__(self,parent=None):
        QtGui.QWidget.__init__(self, parent)
        print "___ - DEBUG ----- CDControlClusters: __init__() "


        # the class global keeping track of the selected image within the sequence:
        #    0 = minimum = the first image in the sequence stack
        self.imageCurrentIndex = 0

        self.theCellTypeColor = QtGui.QColor(QtCore.Qt.green)

        # dict from rgba color to button widget:
        self.theColorButtonDict = dict()

        #
        # QWidget setup (1) - windowing GUI setup for Control Panel:
        #

        self.setWindowTitle("Cluster Editor Window Title")
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

        self.controlEditClusterOfCells = QtGui.QGroupBox("Draw a Cluster of Cells")
        self.controlEditClusterOfCells.setLayout(QtGui.QHBoxLayout())
        self.controlEditClusterOfCells.layout().setContentsMargins(0,0,0,0)
        self.controlEditClusterOfCells.layout().setSpacing(4)
        self.controlEditClusterOfCells.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)











        # ----------------------------------------------------------------
        #
        #   prepare two separate buttons:
        #    one for fill color with a pop-up menu,
        #    and another one to toggle the table of types
        #
# 
#         self.typesGroupBox = QtGui.QGroupBox("PIFF Types")
#         self.typesGroupBox.setLayout(QtGui.QHBoxLayout())
#         self.typesGroupBox.layout().setMargin(2)
#         self.typesGroupBox.layout().setSpacing(4)
#         self.typesGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
# 
#         # the fillColorToolButton is a pop-up menu button,
#         #    Menu defaults are set here and for consistency they *must* coincide
#         #    with the defaults set in the DiagramScene class globals.
#         #
#         self.fillColorToolButton = QtGui.QToolButton()
#         self.fillColorToolButton.setText("Color")
# 
#         self.fillColorToolButton.setIcon( \
#             self.createFloodFillToolButtonIcon( \
#                 ':/icons/floodfill.png', \
#                 QtGui.QColor(QtCore.Qt.green)    )   )
#         self.fillColorToolButton.setIconSize(QtCore.QSize(24, 24))
# 
#         self.fillColorToolButton.setStatusTip("Set the sequence's types")
#         self.fillColorToolButton.setToolTip("Set the sequence's types")
# 
#         self.fillColorToolButton.setPopupMode(QtGui.QToolButton.MenuButtonPopup)
# 
#         # attach a popup menu to the button, with event handler handleFillColorChanged()
#         self.fillColorToolButton.setMenu(  \
#             self.createColorMenu(self.handleFillColorChanged, \
#             QtCore.Qt.green)  )
# 
#         self.fillColorToolButton.menu().setStatusTip("Set the sequence's types (Menu)")
#         self.fillColorToolButton.menu().setToolTip("Set the sequence's types (Menu)")
# 
#         self.fillAction = self.fillColorToolButton.menu().defaultAction()
# 
#         # provide a "slot" function to the button, the event handler is handleFillButtonClicked()
#         self.fillColorToolButton.clicked.connect(self.handleFillButtonClicked)
# 
# 
# 
#         # ------------------------------------------------------------
#         # 2011 - the fillColorToolButton is a button acting as toggle,
#         #   with the pifTableAction used to show/hide the Table of Types window:
#         # Note: PyQt 4.8.6 seems to have problems with assigning the proper key shortcuts
#         #   using mnemonics such as  shortcut=QtGui.QKeySequence.AddTab, so we have to 
#         #   set the shortcut explicitly to "Ctrl+key" ...
#         self.pifTableAction = QtGui.QAction(
#                 QtGui.QIcon(':/icons/regiontable.png'), "Table of Types", self,
#                 shortcut="Ctrl+T", statusTip="Toggle (show/hide) the Table of Types window",
#                 triggered=self.handlePIFTableButton)
# 
#         # add an action to the QGroupBox:
#         lToolButton = QtGui.QToolButton(self)
#         lToolButton.setDefaultAction(self.pifTableAction)
#         lToolButton.setCheckable(self.pifTableAction.isCheckable())
#         lToolButton.setChecked(self.pifTableAction.isChecked())
#         lToolButton.setIcon(self.pifTableAction.icon())
#         lToolButton.setIconSize(QtCore.QSize(24, 24))
#         lToolButton.setToolTip(self.pifTableAction.toolTip())
#         lToolButton.setStatusTip(self.pifTableAction.statusTip())
# 
#         # add all buttons to the QGroupBox:
#         self.typesGroupBox.layout().addWidget(self.fillColorToolButton)
# 
#         self.typesGroupBox.layout().addWidget(lToolButton)





        # ----------------------------------------------------------------
        # place all QGroupBox items in the main controls layout:
        #


        self.theColorSelectionGroupBox = self.createColorGroupBox( \
            self.handleCellTypeColorClicked, self.theCellTypeColor )

        self.imageSequenceControlsMainLayout.addWidget(self.theColorSelectionGroupBox)

        # here comes the main pixel editor:
        self.imageSequenceControlsMainLayout.addWidget(self.controlEditClusterOfCells)

#         self.imageSequenceControlsMainLayout.addWidget(self.areaOrBorderSelectionGroupBox)
#         self.imageSequenceControlsMainLayout.addWidget(self.typesGroupBox)







    # ------------------------------------------------------------
    # 2011 - Mitja: creating a "Color" groupbox produces a QGroupBox item,
    #    and a QButtonGroup, which allows color-picking:
    #
    # ------------------------------------------------------------
    def createColorGroupBox(self, pSlotFunction, pDefaultColor):

        # the QGroupBox widget
        lColorGroupBox = QtGui.QGroupBox("Cell Type")
        lColorGroupBox.setLayout(QtGui.QHBoxLayout())
        lColorGroupBox.layout().setMargin(2)
        lColorGroupBox.layout().setSpacing(0)
        lColorGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                
        for lColor, lName in zip(CDConstants.TypesColors, CDConstants.TypesColorNames):
            # print "lColor =", lColor, "lName =", lName

            lAction = QtGui.QAction(self.createColorIcon(lColor),
                      QtCore.QString(lName), self, triggered=pSlotFunction)

            # set the action's data to be the color:
            lAction.setData(QtGui.QColor(lColor))

            if lColor == pDefaultColor:
                self.addActionToColorGroupBox(lColorGroupBox, lAction, True)
            else:
                self.addActionToColorGroupBox(lColorGroupBox, lAction, False)
        return lColorGroupBox




    # ------------------------------------------------------------------
    # add an action as a button widget to the QGroupBox:
    # ------------------------------------------------------------------
    def addActionToColorGroupBox(self, pGroupBox, pAction, pIsDefault=False):

        lToolButton = QtGui.QToolButton(self)
        lToolButton.setDefaultAction(pAction)
        lToolButton.setCheckable(pAction.isCheckable())
        if (pIsDefault == True):
            lToolButton.setChecked(True)
        else:
            lToolButton.setChecked(False)
        lToolButton.setIcon(pAction.icon())
        lToolButton.setIconSize(QtCore.QSize(16, 16))
        lToolButton.setToolTip(pAction.toolTip())
        lToolButton.setStatusTip(pAction.toolTip() + " Cell Type")
        lToolButton.clearFocus()

        pGroupBox.layout().addWidget(lToolButton)
        
        # also add to global dict of buttons, with their color names as keys:
        self.theColorButtonDict[str(pAction.text())] = lToolButton
        print "pAction.text() =", pAction.text()
        print "self.theColorButtonDict =", self.theColorButtonDict

        # end of def addActionToColorGroupBox(self)
        # ------------------------------------------------------------
       

    # ------------------------------------------------------------
    # 2011 - Mitja: creating a "Color" menu produces a QMenu item,
    #    which allows the creation of a color-picking menu:
    #
    # ------------------------------------------------------------
#     def createColorMenu(self, pSlotFunction, pDefaultColor):
#         lColorMenu = QtGui.QMenu(self)
#         for lColor, lName in zip(CDConstants.TypesColors, CDConstants.TypesColorNames):
#             # print "lColor =", lColor, "lName =", lName
#             lAction = QtGui.QAction(self.createColorIcon(lColor),
#                       QtCore.QString(lName), self, triggered=pSlotFunction)
#             # set the action's data to be the color:
#             lAction.setData(QtGui.QColor(lColor))
#             lColorMenu.addAction(lAction)
#             if lColor == pDefaultColor:
#                 lColorMenu.setDefaultAction(lAction)
#         return lColorMenu

    # ------------------------------------------------------------
    def createColorIcon(self, color):
        pixmap = QtGui.QPixmap(16, 16)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtCore.Qt.NoPen)
        painter.fillRect(QtCore.QRect(0, 0, 16, 16), QtGui.QBrush(QtGui.QColor(color)))
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
        CDConstants.printOut("CDControlClusters - setMaxImageIndex( " +str(pValueInt)+" )", CDConstants.DebugVerbose)











    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "triggered" events
    #    (AKA signals) arriving from the theColorSelectionGroupBox:
    # ------------------------------------------------------------
    def handleCellTypeColorClicked(self):

        # get data from the triggered action:
        self.clickedCellTypeAction = self.sender()
        print self.clickedCellTypeAction
        print dir(self.clickedCellTypeAction)

        # the data received from the action is a QVariant, can be converted to QColor:
        self.chosenCellTypeColor = QtGui.QColor(self.clickedCellTypeAction.data())
        print self.chosenCellTypeColor

        for lKey in self.theColorButtonDict:
            lButton = self.theColorButtonDict[lKey]
            lButton.setChecked(False)
            lButton.setFocus(False)

        for lColor, lName in zip(CDConstants.TypesColors, CDConstants.TypesColorNames):
            # print "lColor =", lColor, "lName =", lName
            if ( QtGui.QColor(lColor).rgba() == self.chosenCellTypeColor.rgba()):
                lButton = self.theColorButtonDict[str(lName)]
                lButton.setChecked(True)
                lButton.setFocus(True)



        # print "self.chosenCellTypeColor is now", self.chosenCellTypeColor, "not", QtGui.QColor(self.chosenCellTypeColor)
       
#         self.fillColorToolButton.setIcon( \
#             self.createFloodFillToolButtonIcon(':/icons/floodfill.png', \
#             QtGui.QColor(self.chosenCellTypeColor)  )  )
#         self.fillColorToolButton.setIconSize(QtCore.QSize(24, 24))

        # propagate the signal upstream, for example to parent objects:
        self.signalSetCurrentTypeColor.emit( QtGui.QColor(self.chosenCellTypeColor) )



    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "clicked" events
    #    (AKA signals) arriving from the fillColorToolButton button:
    # ------------------------------------------------------------
#     def handleFillButtonClicked(self):
#         # there is no change in chosen color when the button is clicked, so
#         #    propagate the signal upstream, for example to parent objects:
#         self.signalSetCurrentTypeColor.emit( QtGui.QColor(self.chosenCellTypeColor) )



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





# end class CDControlClusters(QtGui.QWidget)
# ======================================================================
