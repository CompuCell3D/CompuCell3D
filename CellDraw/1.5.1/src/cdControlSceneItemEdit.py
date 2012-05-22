#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ======================================================================
# 2011 - Mitja: controls for editing items in the Cell Scene, in a box:
#               (a QGroupBox-based control)
# ======================================================================
#
class PIFControlSceneItemEdit(QtGui.QWidget):

    # ------------------------------------------------------------

    def __init__(self,parent=None):

        QtGui.QWidget.__init__(self, parent)

        # ----------------------------------------------------------------
        #
        # QWidget setup (1) - windowing GUI setup for Control Panel:
        #

        self.setWindowTitle("Scene Item Edit Window Title")
        self.itemEditMainLayout = QtGui.QVBoxLayout()
        self.itemEditMainLayout.setMargin(2)
        self.itemEditMainLayout.setSpacing(4)
        self.itemEditMainLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.setLayout(self.itemEditMainLayout)

        # Prepare the font for the radio buttons' caption text:
        lFont = QtGui.QFont()

        # Setting font sizes for Qt widget does NOT work correctly across platforms,
        #   for example the following setPointSize() shows a smaller-than-standard
        #   font on Mac OS X, but it shows a larger-than-standard font on Linux.
        #   Therefore setPointSize() can't be used directly like this:
        # lFont.setPointSize(11)
        lFont.setWeight(QtGui.QFont.Light)


        # create a groupbox for the control layout:
        self.itemEditGroupBox = QtGui.QGroupBox("Item Edit")
        # self.itemEditGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
        # self.itemEditGroupBox.setAutoFillBackground(True)
        self.itemEditGroupBox.setLayout(QtGui.QHBoxLayout())
        self.itemEditGroupBox.layout().setMargin(2)
        self.itemEditGroupBox.layout().setSpacing(4)
        self.itemEditGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)



        # setWindowOpacity seems to work only if it's set after setting WindowFlags and attributes:
        self.setWindowOpacity(0.95)

        # end of def __init__(self,parent=None)
        # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    def populateControlsForSceneItemEdit(self):

        # ----------------------------------------------------------------
        #
        # QWidget setup (2) - place the groupbox in the main control layout:
        #

        # don't create buttons for the QGroupBox, they are added as actions,
        #   by calling addActionToControlsForSceneItemEdit() below:

        # finally add the QGroupBox  to the main layout in the widget:
        self.itemEditMainLayout.addWidget(self.itemEditGroupBox)


        # ----------------------------------------------------------------


    # ------------------------------------------------------------------
    def addActionToControlsForSceneItemEdit(self, pAction):

        # add an action to the QGroupBox:
        lToolButton = QtGui.QToolButton(self)
        lToolButton.setDefaultAction(pAction)
        lToolButton.setCheckable(pAction.isCheckable())
        lToolButton.setChecked(pAction.isChecked())
        lToolButton.setIcon(pAction.icon())
        lToolButton.setIconSize(QtCore.QSize(24, 24))
        lToolButton.setToolTip(pAction.toolTip())
        lToolButton.setStatusTip(pAction.toolTip() + " Scene Item")

        self.itemEditGroupBox.layout().addWidget(lToolButton)

        CDConstants.printOut( "___ - DEBUG ----- PIFControlSceneItemEdit: addActionToControlsForSceneItemEdit("+ str(pAction) + ") done." , CDConstants.DebugTODO )
            

        # end of def addActionToControlsForSceneItemEdit(self)
        # ------------------------------------------------------------
       




# end class PIFControlSceneItemEdit(QtGui.QWidget)
# ======================================================================

