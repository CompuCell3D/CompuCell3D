#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class for drawing an image layer on a QGraphicsScene:
# from cdImageLayer import CDImageLayer

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ======================================================================
# a QWidget-based control
# ======================================================================


# ------------------------------------------------------------
# 2011 - Mitja: to control the "picking mode" for the input image,
#   we add a set of radio-buttons inside a box:
# ------------------------------------------------------------
# note: this class emits four signals:
#
#         self.emit(QtCore.SIGNAL("inputImagePickingModeChangedSignal()"))
#         self.emit(QtCore.SIGNAL("inputImageOpacityChangedSignal()"))
#         self.emit(QtCore.SIGNAL("fuzzyPickTresholdChangedSignal()"))
#
#         signalImageScaleZoomHasChanged = QtCore.pyqtSignal(str)
#
class CDControlInputImage(QtGui.QWidget):

    # ------------------------------------------------------------

    signalImageScaleZoomHasChanged = QtCore.pyqtSignal(str)

    # ------------------------------------------------------------

    def __init__(self,parent=None):
        QtGui.QWidget.__init__(self, parent)

        # the class global keeping track of the selected picking mode:
        #    1 = Color Pick = CDConstants.ImageModePickColor
        #    2 = Freehand Draw = CDConstants.ImageModeDrawFreehand
        #    3 = Polygon Draw = CDConstants.ImageModeDrawPolygon
        self.theInputImagePickingMode = CDConstants.ImageModeDrawFreehand

        # the class global keeping track of the required opacity:
        #      0 = minimum = the image is completely transparent (invisible)
        #    100 = maximum = the image is completely opaque
        self.theImageOpacity = 100

        # the class global keeping track of the fuzzy pick treshold:
        #      0 = minimum = pick only the seed color
        #    100 = maximum = pick everything in the image
        self.theFuzzyPickTreshold = 2

        # the class global keeping track of the current scale/zoom value:
        self.theScaleZoom = "100%"

        #
        # QWidget setup (1) - windowing GUI setup for Control Panel:
        #

        self.setWindowTitle("Image Layer Window Title")
        # QVBoxLayout layout lines up widgets vertically:
        self.imageControlsMainLayout = QtGui.QVBoxLayout()
        self.imageControlsMainLayout.setContentsMargins(0,0,0,0)
        self.imageControlsMainLayout.setSpacing(0)
        self.imageControlsMainLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
#         self.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
#         self.setAutoFillBackground(True)


        self.setLayout(self.imageControlsMainLayout)


        # Prepare three radio buttons, one for each distinct "picking mode"
        #   and the Color Pick mode is set as the default.

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
        # QWidget setup (0) - define a simple layout for drawing + zoom:
        #

        aSimpleQHBoxLayout = QtGui.QHBoxLayout()
        aSimpleQHBoxLayout.setContentsMargins(0,0,0,0)
        aSimpleQHBoxLayout.setSpacing(4)
        aSimpleQHBoxLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)



        # ----------------------------------------------------------------
        #
        # QWidget setup (1) - add controls for Drawing in the Image:
        #
        print "___ - DEBUG ----- CDControlInputImage: populateControlPanel() 2"

        self.drawingGroupBox = QtGui.QGroupBox("Drawing tools")
#         self.drawingGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
#         self.drawingGroupBox.setAutoFillBackground(True)
        self.drawingGroupBox.setLayout(QtGui.QHBoxLayout())
        self.drawingGroupBox.layout().setContentsMargins(2,2,2,2)
        self.drawingGroupBox.layout().setSpacing(4)
        self.drawingGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.pickFreehandRegionButton = QtGui.QToolButton()
        self.pickFreehandRegionButton.setCheckable(True)
        self.pickFreehandRegionButton.setChecked(True)
        self.pickFreehandRegionButton.setIcon(QtGui.QIcon(':/icons/drawPencil.png'))
        self.pickFreehandRegionButton.setIconSize(QtCore.QSize(24, 24))
        self.pickFreehandRegionButton.setStatusTip("Freehand Draw: draw over the input image to create a new region in the Cell Scene")
        self.pickFreehandRegionButton.setToolTip("Freehand Draw")
        # self.layout().addWidget(self.pickFreehandRegionButton)

        self.pickPolygonRegionButton = QtGui.QToolButton()
        self.pickPolygonRegionButton.setCheckable(True)
        self.pickPolygonRegionButton.setChecked(False)
        self.pickPolygonRegionButton.setIcon(QtGui.QIcon(':/icons/drawPolygon.png'))
        self.pickPolygonRegionButton.setIconSize(QtCore.QSize(24, 24))
        self.pickPolygonRegionButton.setStatusTip("Polygon Draw: draw over the input image to create a new region in the Cell Scene")
        self.pickPolygonRegionButton.setToolTip("Polygon Draw")

        self.drawingGroupBox.layout().addWidget(self.pickFreehandRegionButton)
        self.drawingGroupBox.layout().addWidget(self.pickPolygonRegionButton)





        # ----------------------------------------------------------------
        #
        # QWidget setup (4) - "Scale/Zoom" QGroupBox:

        self.scaleZoomGroupBox = QtGui.QGroupBox("Scale/Zoom")
        self.scaleZoomGroupBox.setLayout(QtGui.QHBoxLayout())
        self.scaleZoomGroupBox.layout().setContentsMargins(2,2,2,2)
        self.scaleZoomGroupBox.layout().setSpacing(4)
        self.scaleZoomGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # a "combo box" pop-up menu to select the Scale/Zoom factor:
        self.imageScaleCombo = QtGui.QComboBox()
        self.imageScaleCombo.addItems(["50%", "75%", "100%", "125%", "150%", "200%", "250%", "300%", "400%", "500%", "1000%", "2000%", "4000%"])
        self.imageScaleCombo.setCurrentIndex(2)
        self.imageScaleCombo.clearFocus()
        self.imageScaleCombo.setStatusTip("Zoom the input image")
        self.imageScaleCombo.setToolTip("Zoom input image")
       
        # call handleScaleZoomChanged() when imageScaleCombo changes index:
        self.imageScaleCombo.currentIndexChanged[str].connect(self.handleScaleZoomChanged)

        # add the combo box to the QGroupBox:
        self.scaleZoomGroupBox.layout().addWidget(self.imageScaleCombo)





        # ----------------------------------------------------------------
        #
        # QWidget setup (0) - add the simple layout for drawing + zoom
        #                       to the main layout:
        #
       
        aSimpleQHBoxLayout.addWidget(self.drawingGroupBox)

        aSimpleQHBoxLayout.addWidget(self.scaleZoomGroupBox)

        self.imageControlsMainLayout.addLayout(aSimpleQHBoxLayout)






        # ----------------------------------------------------------------
        #
        # QWidget setup (2) - add controls for transparency/displaying the Image:
        #
        print "___ - DEBUG ----- CDControlInputImage: populateControlPanel() 2"

        self.displayingImageGroupBox = QtGui.QGroupBox("Image opacity")
#         self.displayingImageGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
#         self.displayingImageGroupBox.setAutoFillBackground(True)
        self.displayingImageGroupBox.setLayout(QtGui.QHBoxLayout())
        self.displayingImageGroupBox.layout().setContentsMargins(0,0,0,0)
        self.displayingImageGroupBox.layout().setSpacing(4)
        self.displayingImageGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # 2011 - Mitja: a QSlider to control the input image's translucency/opacity:
        self.imageOpacitySlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.imageOpacitySlider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.imageOpacitySlider.setTickPosition(QtGui.QSlider.NoTicks)
        self.imageOpacitySlider.setSingleStep(1)
        self.imageOpacitySlider.setMinimum(0)
        self.imageOpacitySlider.setMaximum(100)
        self.imageOpacitySlider.setValue(100)
        self.imageOpacitySlider.setStatusTip("Image Opacity: fully transparent image when the slider is on the left side, fully opaque on the right.")
        self.imageOpacitySlider.setToolTip("Image Opacity")
        self.imageOpacitySlider.valueChanged.connect(self.imageOpacityChanged)

        self.imageOpacityLabel = QtGui.QLabel()
        self.imageOpacityLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lFloatValue = (float(self.imageOpacitySlider.value()) * 0.01)
        lStrValue = str(lFloatValue)
        self.imageOpacityLabel.setText(lStrValue)
        self.imageOpacityLabel.setMinimumSize( QtCore.QSize(48, 24) )
        self.imageOpacityLabel.setContentsMargins(0,0,0,0)
        self.imageOpacityLabel.setFont(lFont)

        self.displayingImageGroupBox.layout().addWidget(self.imageOpacitySlider)
        self.displayingImageGroupBox.layout().addWidget(self.imageOpacityLabel)

        self.imageControlsMainLayout.addWidget(self.displayingImageGroupBox)


        # ----------------------------------------------------------------
        #
        # QWidget setup (3) - add controls for Color picking in the Image:
        #
        print "___ - DEBUG ----- CDControlInputImage: populateControlPanel() 1"

        self.colorPickingGroupBox = QtGui.QGroupBox("Color pick")
#         self.colorPickingGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
#         self.colorPickingGroupBox.setAutoFillBackground(True)
        self.colorPickingGroupBox.setLayout(QtGui.QHBoxLayout())
        self.colorPickingGroupBox.layout().setContentsMargins(2,2,2,2)
        self.colorPickingGroupBox.layout().setSpacing(4)
        self.colorPickingGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)


        # 2011 - Mitja: a QSlider to control the input image's translucency/opacity:
        self.fuzzyPickThresholdSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.fuzzyPickThresholdSlider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.fuzzyPickThresholdSlider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.fuzzyPickThresholdSlider.setSingleStep(1)
        self.fuzzyPickThresholdSlider.setMinimum(0)
        self.fuzzyPickThresholdSlider.setMaximum(100)
        self.fuzzyPickThresholdSlider.setValue(2)
        self.fuzzyPickThresholdSlider.setStatusTip("Color Pick Threshold: slider to the left side for single color pick, slider to the right to pick everything.")
        self.fuzzyPickThresholdSlider.setToolTip("Color Pick Threshold")
        self.fuzzyPickThresholdSlider.valueChanged.connect(self.fuzzyPickThresholdChanged)
       
        self.fuzzyPickThresholdLabel = QtGui.QLabel()
        self.fuzzyPickThresholdLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lFloatValue = (float(self.fuzzyPickThresholdSlider.value()) * 0.01) * \
            (float(self.fuzzyPickThresholdSlider.value()) * 0.01)
        lStrValue = str(lFloatValue)
        self.fuzzyPickThresholdLabel.setText(lStrValue)
        self.fuzzyPickThresholdLabel.setMinimumSize( QtCore.QSize(58, 24) )
        self.fuzzyPickThresholdLabel.setContentsMargins(0,0,0,0)
        self.fuzzyPickThresholdLabel.setFont(lFont)

        self.pickColorRegionButton = QtGui.QToolButton()
        self.pickColorRegionButton.setCheckable(True)
        self.pickColorRegionButton.setChecked(False)
        self.pickColorRegionButton.setIcon(QtGui.QIcon(':/icons/pickColor.png'))
        self.pickColorRegionButton.setIconSize(QtCore.QSize(24, 24))
        self.pickColorRegionButton.setStatusTip("Color Pick: click on a color in the input image to create a new region in the Cell Scene")
        self.pickColorRegionButton.setToolTip("Color Pick")
        # self.layout().addWidget(self.pickColorRegionButton)

        self.colorPickingGroupBox.layout().addWidget(self.pickColorRegionButton)
        self.colorPickingGroupBox.layout().addWidget(self.fuzzyPickThresholdSlider)
        self.colorPickingGroupBox.layout().addWidget(self.fuzzyPickThresholdLabel)

        self.imageControlsMainLayout.addWidget(self.colorPickingGroupBox)






        # ----------------------------------------------------------------
        #
        # QWidget setup (5) - add logical QButtonGroup:
        #
        # ----------------------------------------------------------------


        # 2010 - Mitja: "picking" QButtonGroup, a logical container to make buttons mutually exclusive:
        self.pickingModeGroup = QtGui.QButtonGroup()
        self.pickingModeGroup.addButton(self.pickColorRegionButton, CDConstants.ImageModePickColor)
        self.pickingModeGroup.addButton(self.pickFreehandRegionButton, CDConstants.ImageModeDrawFreehand)
        self.pickingModeGroup.addButton(self.pickPolygonRegionButton, CDConstants.ImageModeDrawPolygon)
        # call handleImageButtonGroupClicked() every time a button is clicked in the "pickingModeGroup"
        self.pickingModeGroup.buttonClicked[int].connect(self.handleImageButtonGroupClicked)







        # setWindowOpacity seems to work only if it's set after setting WindowFlags and attributes:
        self.setWindowOpacity(0.95)






    def handleImageButtonGroupClicked(self, pChecked):
        if self.pickColorRegionButton.isChecked():
            lPickingMode = CDConstants.ImageModePickColor
        elif self.pickFreehandRegionButton.isChecked():
            lPickingMode = CDConstants.ImageModeDrawFreehand
        elif self.pickPolygonRegionButton.isChecked():
            lPickingMode = CDConstants.ImageModeDrawPolygon
        if lPickingMode is not self.theInputImagePickingMode:
            self.theInputImagePickingMode = lPickingMode
            # print "the picking mode is =", self.theInputImagePickingMode
            # propagate the signal upstream, for example to parent objects:
            self.emit(QtCore.SIGNAL("inputImagePickingModeChangedSignal()"))

    def imageOpacityChanged(self, pValue):
        self.theImageOpacity = pValue
        lFloatValue = (float(pValue) * 0.01)
        lStrValue = str(lFloatValue)
        self.imageOpacityLabel.setText(lStrValue)
        # propagate the signal upstream, for example to parent objects:
        self.emit(QtCore.SIGNAL("inputImageOpacityChangedSignal()"))

    def fuzzyPickThresholdChanged(self, pValue):
        self.theFuzzyPickTreshold = pValue
        lFloatValue = (float(pValue) * 0.01) * (float(pValue) * 0.01)
        lStrValue = str(lFloatValue)
        self.fuzzyPickThresholdLabel.setText(lStrValue)
        # propagate the signal upstream, for example to parent objects:
        self.emit(QtCore.SIGNAL("fuzzyPickTresholdChangedSignal()"))

    def imageOpacityChanged(self, pValue):
        self.theImageOpacity = pValue
        lFloatValue = (float(pValue) * 0.01)
        lStrValue = str(lFloatValue)
        self.imageOpacityLabel.setText(lStrValue)
        # propagate the signal upstream, for example to parent objects:
        self.emit(QtCore.SIGNAL("inputImageOpacityChangedSignal()"))

    def handleScaleZoomChanged(self, pValueString):
        print "the image scale/zoom is =", pValueString
        lScaleZoom = pValueString
        if lScaleZoom is not self.theScaleZoom:
            self.theScaleZoom = lScaleZoom
            # print "the layer mode is =", self.theLayerMode
            # propagate the signal upstream, for example to parent objects:
            self.signalImageScaleZoomHasChanged.emit(self.theScaleZoom)



# end class CDControlInputImage(QtGui.QWidget)
# ======================================================================

