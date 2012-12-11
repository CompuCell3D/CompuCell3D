#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ======================================================================
# 2011 - Mitja: user controls for setting types of regions and cells:
#               (a QGroupBox-based control)
# ======================================================================
# note: this class emits two signals:
#
#          signalSetCurrentTypeColor = QtCore.pyqtSignal(int)
#
#          signalForPIFFTableToggle = QtCore.pyqtSignal(str)
#
class CDControlTypes(QtGui.QWidget):

    # ------------------------------------------------------------

    # the signal used to communicate color menu changes:
    signalSetCurrentTypeColor = QtCore.pyqtSignal(QtCore.QVariant)

    # the signal used to toggle visibility of the Table of Types:
    signalForPIFFTableToggle = QtCore.pyqtSignal(str)

    # ------------------------------------------------------------

    def __init__(self,parent=None):

        QtGui.QWidget.__init__(self, parent)

#         CDConstants.TypesColorNames = [QtCore.Qt.green, QtCore.Qt.blue, QtCore.Qt.red, \
#                        QtCore.Qt.darkYellow, QtCore.Qt.lightGray, QtCore.Qt.magenta, \
#                        QtCore.Qt.darkBlue, QtCore.Qt.cyan, QtCore.Qt.darkGreen, QtCore.Qt.white]
#         CDConstants.TypesColors = ["green", "blue", "red", "darkYellow", "lightGray", \
#                       "magenta", "darkBlue", "cyan", "darkGreen", "white"]

        # 2011 - Mitja: pick a color for the PIFF region or cell,
        #    with a variable keeping track of what is to be used:
        self.chosenMenuColor = 0

        # ----------------------------------------------------------------
        #
        # QWidget setup (1) - windowing GUI setup for Types of Regions or Cells controls:
        #

        self.setWindowTitle("Types of Regions or Cells Window Title")
        self.typesMainLayout = QtGui.QVBoxLayout()
        self.typesMainLayout.setMargin(2)
        self.typesMainLayout.setSpacing(4)
        self.typesMainLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.setLayout(self.typesMainLayout)

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
        # QWidget setup (2) - prepare two separate buttons:
        #    one for fill color with a pop-up menu,
        #    and another one to toggle the table of types
        #

        self.typesGroupBox = QtGui.QGroupBox("PIFF Types")
        # self.typesGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
        # self.typesGroupBox.setAutoFillBackground(True)
        self.typesGroupBox.setLayout(QtGui.QHBoxLayout())
        self.typesGroupBox.layout().setMargin(2)
        self.typesGroupBox.layout().setSpacing(4)
        self.typesGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # create buttons for the QGroupBox:
#
#         # end of def __init__(self,parent=None)
#         # ------------------------------------------------------------------
#
#     # ------------------------------------------------------------------
#     def populateControlsForTypes(self):

        # button for Item == Region:

        # ------------------------------------------------------------
        # 2010 - the fillColorToolButton is a pop-up menu button,
        #    Menu defaults are set here and for consistency they *must* coincide
        #      with the defaults set in the DiagramScene class globals.
        #
        self.fillColorToolButton = QtGui.QToolButton()
        self.fillColorToolButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.fillColorToolButton.setText("Type")

        self.fillColorToolButton.setIcon( \
            self.createFloodFillToolButtonIcon( \
                ':/icons/floodfill.png', \
                QtGui.QColor(QtCore.Qt.green)    )   )
        self.fillColorToolButton.setIconSize(QtCore.QSize(24, 24))

        self.fillColorToolButton.setStatusTip("Set the selected PIFF Region\'s or Cell's color")
        self.fillColorToolButton.setToolTip("Set PIFF Region or Cell Color")

        self.fillColorToolButton.setPopupMode(QtGui.QToolButton.MenuButtonPopup)

        # attach a popup menu to the button, with event handler handleFillColorChanged()
        self.fillColorToolButton.setMenu(  \
            self.createColorMenu(self.handleFillColorChanged, \
            QtCore.Qt.green)  )

        self.fillColorToolButton.menu().setStatusTip("Set the selected PIFF Region\'s or Cell's color (Menu)")
        self.fillColorToolButton.menu().setToolTip("Set PIFF Region or Cell Color(Menu)")

        self.fillAction = self.fillColorToolButton.menu().defaultAction()

        # provide a "slot" function to the button, the event handler is handleFillButtonClicked()
        self.fillColorToolButton.clicked.connect(self.handleFillButtonClicked)



        # ------------------------------------------------------------
        # 2011 - the fillColorToolButton is a button acting as toggle,
        #   with the pifTableAction used to show/hide the Table of Types window:
        # Note: PyQt 4.8.6 seems to have problems with assigning the proper key shortcuts
        #   using mnemonics such as  shortcut=QtGui.QKeySequence.AddTab, so we have to 
        #   set the shortcut explicitly to "Ctrl+key" ...
        self.pifTableAction = QtGui.QAction(self)

        CDConstants.printOut("___ - DEBUG ----- CDControlTypes.__init__() -- self.pifTableAction = QtGui.QAction() ====== "+str(self.pifTableAction), CDConstants.DebugExcessive )

#         self.pifTableAction = QtGui.QAction( \
#                 QtGui.QIcon(':/icons/regiontable.png'), "Types Editor", self, \
#                 shortcut="Ctrl+T", statusTip="Toggle (show/hide) the Types Editor", \
#                 triggered=self.handlePIFTableButton)

        # a new button to show the image sequence layer:
        self.pifTableAction.setCheckable(True)
        self.pifTableAction.setChecked(False)
        self.pifTableAction.setShortcut("Ctrl+T")
        self.pifTableAction.setIcon(QtGui.QIcon(':/icons/regiontable.png'))
#        self.pifTableAction.setIconSize(QtCore.QSize(24, 24))
        self.pifTableAction.setIconText("Types")
        self.pifTableAction.setToolTip("Types Editor")
        self.pifTableAction.setStatusTip("Toggle (show/hide) the Types Editor")
        self.pifTableAction.triggered[bool].connect(self.handlePIFTableButton)
#         self.imageSequenceAction.hide()



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
        self.typesGroupBox.layout().addWidget(self.fillColorToolButton)

        self.typesGroupBox.layout().addWidget(lToolButton)

        # finally add the QGroupBox  to the main layout in the widget:
        self.typesMainLayout.addWidget(self.typesGroupBox)

        # setWindowOpacity seems to work only if it's set after setting WindowFlags and attributes:
        self.setWindowOpacity(0.95)





    # ------------------------------------------------------------
    # 2010 - Mitja: creating a "Color" menu produces a QMenu item,
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
        # CDConstants.printOut( " "+str( "OOOOOOOOOOOOOO in createFloodFillToolButtonIcon() pColor =", pColor )+" ", CDConstants.DebugTODO )
        lPainter.fillRect(QtCore.QRect(0, 60, 80, 80), pColor)
        lPainter.drawPixmap(lTarget, lImage, lSource)
        lPainter.end()
        return QtGui.QIcon(lPixmap)



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
    # 2010 Mitja - set the menu where the PIFF Types Table ought to appear:
    # ------------------------------------------------------------
    def setMenuForTableAction(self, pMenu):
        pMenu.addAction(self.pifTableAction)




# end class CDControlTypes(QtGui.QWidget)
# ======================================================================

# Local Variables:
# coding: US-ASCII
# End:
