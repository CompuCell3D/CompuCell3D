#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ======================================================================
# 2011 - Mitja: toggle cell / region drawing, two radio-buttons in a box:
#               (a QGroupBox-based control)
# ======================================================================
# note: this class emits a signal:
#
#       signalSetRegionOrCell = QtCore.pyqtSignal(int)
#
class ControlRegionOrCell(QtGui.QWidget):

    # ------------------------------------------------------------

    # the signal used to communicate toggle changes:
    signalSetRegionOrCell = QtCore.pyqtSignal(int)

    # ------------------------------------------------------------

    def __init__(self,parent=None):

        QtGui.QWidget.__init__(self, parent)

        # 2011 - Mitja: toggle between drawing cells vs. regions,
        #    with a constant keeping track of what is to be used:
        #    0 = Cell Draw = CDConstants.ItsaCellConst
        #    1 = Region Draw = CDConstants.ItsaRegionConst
        self.drawRegionOrCell = CDConstants.ItsaRegionConst

        # ----------------------------------------------------------------
        #
        # QWidget setup (1) - windowing GUI setup for Control Panel:
        #

        self.setWindowTitle("Draw Toggle Window Title")
        self.regionOrCellMainLayout = QtGui.QVBoxLayout()
        self.regionOrCellMainLayout.setMargin(0)
        self.regionOrCellMainLayout.setSpacing(0)
        self.regionOrCellMainLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.setLayout(self.regionOrCellMainLayout)

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
        # QWidget setup (2) - prepare two radio buttons, one for each
        #   distinct "drawing mode" and the ItsaRegionConst is set as the default.
        #

        self.regionOrCellGroupBox = QtGui.QGroupBox("Item is...")
        # self.regionOrCellGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
        # self.regionOrCellGroupBox.setAutoFillBackground(True)
        self.regionOrCellGroupBox.setLayout(QtGui.QHBoxLayout())
        self.regionOrCellGroupBox.layout().setMargin(2)
        self.regionOrCellGroupBox.layout().setSpacing(4)
        self.regionOrCellGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # create buttons for the QGroupBox:

        # button for Item == Region:
        self.pickRegionDrawingButton = QtGui.QToolButton()
        self.pickRegionDrawingButton.setText("Region")
        self.pickRegionDrawingButton.setCheckable(True)
        self.pickRegionDrawingButton.setChecked(True)
        self.pickRegionDrawingButton.setIcon(QtGui.QIcon(':/icons/itemIsRegion.png'))
        self.pickRegionDrawingButton.setIconSize(QtCore.QSize(24, 24))
        # for region items, use darkMagenta pen:
#         self.pickRegionDrawingButton.setPalette( \
#             QtGui.QPalette(QtGui.QColor(QtCore.Qt.darkMagenta)) )
#         self.pickRegionDrawingButton.setAutoFillBackground(True)
        self.pickRegionDrawingButton.setFont(lFont)
        self.pickRegionDrawingButton.setToolTip("Item is a Region")
        self.pickRegionDrawingButton.setStatusTip("Item is a Region: set the selected item in the Cell Scene to be a region of Cells")

        # button for Item == Cell:
        self.pickCellDrawingButton = QtGui.QToolButton()
        self.pickCellDrawingButton.setText("Cell")
        self.pickCellDrawingButton.setCheckable(True)
        self.pickCellDrawingButton.setChecked(False)
        self.pickCellDrawingButton.setIcon(QtGui.QIcon(':/icons/itemIsCell.png'))
        self.pickCellDrawingButton.setIconSize(QtCore.QSize(24, 24))
        # for cell items, use orange "#FF9900" or (255, 153, 0) :
#         self.pickCellDrawingButton.setPalette( \
#             QtGui.QPalette(QtGui.QColor(255, 153, 0)) )
#         self.pickCellDrawingButton.setAutoFillBackground(True)
        self.pickCellDrawingButton.setFont(lFont)
        self.pickCellDrawingButton.setToolTip("Item is a Cell")
        self.pickCellDrawingButton.setStatusTip("Item is a Cell: set the selected item in the Cell Scene to be a single Cell")
        # 2011 - Mitja: since the self.pickCellDrawingButton button is used
        # TODO FIX: to set items to be individual cells, it's initially not enabled, since its functionality is BUGGY:
        self.pickCellDrawingButton.setEnabled(True)

        # add all buttons to the QGroupBox:
        self.regionOrCellGroupBox.layout().addWidget(self.pickRegionDrawingButton)
        self.regionOrCellGroupBox.layout().addWidget(self.pickCellDrawingButton)       
        # finally add the QGroupBox  to the main layout in the widget:
        self.regionOrCellMainLayout.addWidget(self.regionOrCellGroupBox)


        # ----------------------------------------------------------------
        #
        # QWidget setup (3) - "Layer Selection" QButtonGroup,
        #    a *logical* container to make buttons mutually exclusive:

        self.theButtonGroupForRegionOrCell = QtGui.QButtonGroup()
        self.theButtonGroupForRegionOrCell.addButton(self.pickRegionDrawingButton, CDConstants.ItsaRegionConst)
        self.theButtonGroupForRegionOrCell.addButton(self.pickCellDrawingButton, CDConstants.ItsaCellConst)

        # call handleRadioButtonClick() every time a button is clicked in the "theButtonGroupForRegionOrCell"
        self.theButtonGroupForRegionOrCell.buttonClicked[int].connect( \
            self.handleRadioButtonClick)


        # setWindowOpacity seems to work only if it's set after setting WindowFlags and attributes:
        self.setWindowOpacity(0.95)




    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "toggled" events
    #    (AKA signals) arriving from pickRegionDrawingButton:
    # ------------------------------------------------------------
    def handleRadioButtonClick(self, pChecked):
        if self.pickCellDrawingButton.isChecked():
            lWhatToDraw = CDConstants.ItsaCellConst
            print "ItsaCellConst, lWhatToDraw is now", lWhatToDraw
        elif self.pickRegionDrawingButton.isChecked():
            lWhatToDraw = CDConstants.ItsaRegionConst
            print "ItsaRegionConst, lWhatToDraw is now", lWhatToDraw

        self.drawRegionOrCell = lWhatToDraw
        print "now drawing regions or cells:", self.drawRegionOrCell
        # propagate the signal upstream, for example to parent objects:
        self.signalSetRegionOrCell.emit(self.drawRegionOrCell)



# end class ControlRegionOrCell(QtGui.QWidget)
# ======================================================================

