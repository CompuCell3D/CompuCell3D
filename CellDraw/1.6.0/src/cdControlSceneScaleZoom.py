#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ======================================================================
# 2011 - Mitja: set scale/zoom with a combo box (pop-up menu) :
#               (inside a box, a QGroupBox-based control)
# ======================================================================
# note: this class emits one signal:
#
#         signalScaleZoomHasChanged = QtCore.pyqtSignal(str)
#
class CDControlSceneScaleZoom(QtGui.QWidget):

    # ------------------------------------------------------------

    signalScaleZoomHasChanged = QtCore.pyqtSignal(str)

    # ------------------------------------------------------------
    def __init__(self,parent=None):
        QtGui.QWidget.__init__(self, parent)

        # the class global keeping track of the current scale/zoom value:
        self.theScaleZoom = "100%"

        # ----------------------------------------------------------------
        #
        # QWidget setup (1) - windowing GUI setup for Scene Scale/Zoom controls:
        #

        self.setWindowTitle("Scene Scale/Zoom Window Title")
        # QVBoxLayout layout lines up widgets vertically:
        self.sceneScaleZoomMainLayout = QtGui.QVBoxLayout()
        self.sceneScaleZoomMainLayout.setMargin(2)
        self.sceneScaleZoomMainLayout.setSpacing(4)
        self.sceneScaleZoomMainLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
#         self.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
#         self.setAutoFillBackground(True)
        self.setLayout(self.sceneScaleZoomMainLayout)

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
        # QWidget setup (2) - "Scale/Zoom" QGroupBox:

        self.scaleZoomGroupBox = QtGui.QGroupBox("Scale/Zoom")
        #         self.scaleZoomGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
        #         self.scaleZoomGroupBox.setAutoFillBackground(True)
        self.scaleZoomGroupBox.setLayout(QtGui.QHBoxLayout())
        self.scaleZoomGroupBox.layout().setMargin(2)
        self.scaleZoomGroupBox.layout().setSpacing(4)
        self.scaleZoomGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # a "combo box" pop-up menu to select the Scale/Zoom factor:
        self.sceneScaleCombo = QtGui.QComboBox()
        self.sceneScaleCombo.addItems(["50%", "75%", "100%", "125%", "150%", "200%", "250%", "300%", "400%", "500%", "1000%", "2000%", "4000%"])
        self.sceneScaleCombo.setCurrentIndex(2)
        self.sceneScaleCombo.clearFocus()
        self.sceneScaleCombo.setStatusTip("Zoom the Cell Scene view")
        self.sceneScaleCombo.setToolTip("Zoom Cell Scene")
       
        # call handleScaleZoomChanged() when sceneScaleCombo changes index:
        self.sceneScaleCombo.currentIndexChanged[str].connect(self.handleScaleZoomChanged)

        # add the combo box to the QGroupBox:
        self.scaleZoomGroupBox.layout().addWidget(self.sceneScaleCombo)
        # finally add the QGroupBox  to the main layout in the widget:
        self.sceneScaleZoomMainLayout.addWidget(self.scaleZoomGroupBox)


        # setWindowOpacity seems to work only if it's set after setting WindowFlags and attributes:
        self.setWindowOpacity(0.95)




    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "currentIndexChanged" events
    #    (AKA signals) arriving from sceneScaleCombo:
    # ------------------------------------------------------------
    def handleScaleZoomChanged(self, pValueString):

        CDConstants.printOut( "the requested Scene scale/zoom is = "+str(pValueString), CDConstants.DebugTODO )
        lScaleZoom = pValueString
        if lScaleZoom != self.theScaleZoom:
            self.theScaleZoom = lScaleZoom
            CDConstants.printOut( "the new Scene scale/zoom will be = "+str(self.theScaleZoom), CDConstants.DebugTODO )
            # propagate the signal upstream, for example to parent objects:
            self.signalScaleZoomHasChanged.emit(self.theScaleZoom)



# end class CDControlSceneScaleZoom(QtGui.QWidget)
# ======================================================================
