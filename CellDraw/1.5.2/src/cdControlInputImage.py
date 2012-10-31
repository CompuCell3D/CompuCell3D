#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

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
        #    0 = Color Pick = CDConstants.ImageModePickColor
        #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
        #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon
        #    3 = Extract Cells = CDConstants.ImageModeExtractCells
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
        self.imageControlsMainLayout.setSpacing(4)
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
        CDConstants.printOut( "___ - DEBUG ----- CDControlInputImage.__init__() 1", CDConstants.DebugTODO )

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
        CDConstants.printOut( "___ - DEBUG ----- CDControlInputImage.__init__() 2", CDConstants.DebugTODO )

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
        CDConstants.printOut( "___ - DEBUG ----- CDControlInputImage.__init__() 3", CDConstants.DebugTODO )

        self.colorRegionPickingGroupBox = QtGui.QGroupBox("Color Region Pick")
#         self.colorRegionPickingGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
#         self.colorRegionPickingGroupBox.setAutoFillBackground(True)
        self.colorRegionPickingGroupBox.setLayout(QtGui.QHBoxLayout())
        self.colorRegionPickingGroupBox.layout().setContentsMargins(2,2,2,2)
        self.colorRegionPickingGroupBox.layout().setSpacing(4)
        self.colorRegionPickingGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)


        # 2011 - Mitja: a QSlider to control the input image's translucency/opacity:
        self.fuzzyPickThresholdSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.fuzzyPickThresholdSlider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.fuzzyPickThresholdSlider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.fuzzyPickThresholdSlider.setSingleStep(1)
        self.fuzzyPickThresholdSlider.setMinimum(0)
        self.fuzzyPickThresholdSlider.setMaximum(100)
        self.fuzzyPickThresholdSlider.setValue(2)
        self.fuzzyPickThresholdSlider.setStatusTip("Color Region Pick Threshold: slider to the left side for single color pick, slider to the right to pick everything.")
        self.fuzzyPickThresholdSlider.setToolTip("Color Region Pick Threshold")
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
        self.pickColorRegionButton.setStatusTip("Single Color Pick: click on a color in the input image to create a new region in the Cell Scene")
        self.pickColorRegionButton.setToolTip("Single Color Pick")
        # self.layout().addWidget(self.pickColorRegionButton)


        self.colorRegionPickingGroupBox.layout().addWidget(self.pickColorRegionButton)
        self.colorRegionPickingGroupBox.layout().addWidget(self.fuzzyPickThresholdSlider)
        self.colorRegionPickingGroupBox.layout().addWidget(self.fuzzyPickThresholdLabel)

        self.imageControlsMainLayout.addWidget(self.colorRegionPickingGroupBox)





        # ----------------------------------------------------------------
        #
        # QWidget setup (4) - add controls for extracting cells by colors in the Image:
        #
        CDConstants.printOut( "___ - DEBUG ----- CDControlInputImage.__init__() 4", CDConstants.DebugTODO )


        self.extractCellsGroupBox = QtGui.QGroupBox("Extract Cells by Color")
        self.extractCellsGroupBox.setLayout(QtGui.QHBoxLayout())
        self.extractCellsGroupBox.layout().setContentsMargins(2,2,2,2)
        self.extractCellsGroupBox.layout().setSpacing(4)
        self.extractCellsGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.pickExtractCellsRegionButton = QtGui.QToolButton()
        self.pickExtractCellsRegionButton.setCheckable(True)
        self.pickExtractCellsRegionButton.setChecked(False)
        self.pickExtractCellsRegionButton.setIcon(QtGui.QIcon(':/icons/pickExtractCells.png'))
        self.pickExtractCellsRegionButton.setIconSize(QtCore.QSize(24, 24))
        self.pickExtractCellsRegionButton.setStatusTip("Extract Cells by Color: click on a color in the input image to extract that color as cells in the Cell Scene")
        self.pickExtractCellsRegionButton.setToolTip("Extract Cells by Color")

        self.extractCellsGroupBox.layout().addWidget(self.pickExtractCellsRegionButton)

        self.imageControlsMainLayout.addWidget(self.extractCellsGroupBox)



        # ----------------------------------------------------------------
        #
        # QWidget setup - add a QGroupBox for tracking mouse position etc.:
        #
        self.informationGroupBox = QtGui.QGroupBox("Information")
        #         self.informationGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
        #         self.informationGroupBox.setAutoFillBackground(True)
        self.informationGroupBox.setLayout(QtGui.QVBoxLayout())
        self.informationGroupBox.layout().setMargin(2)
        self.informationGroupBox.layout().setSpacing(4)
        self.informationGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)


        lFont = QtGui.QFont()
        lFont.setWeight(QtGui.QFont.Light)

        sceneItemLayerLayout = QtGui.QHBoxLayout()
        sceneItemLayerLayout.setMargin(2)
        sceneItemLayerLayout.setSpacing(4)
        sceneItemLayerLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

#         sceneItemLayerLayout.addWidget(self.createRegionShapeButton("Ellipse", CDControlPanel.PathConst))
# 
#         sceneItemLayerLayout.addWidget(self.createRegionShapeButton("Rectangle", CDControlPanel.RectangleConst))

        sceneItemLayerLayout.addSpacing(40)

#         sceneItemLayerLayout.addWidget(self.createRegionShapeButton("TenByTenBox", CDControlPanel.TenByTenBoxConst))
# 
#         sceneItemLayerLayout.addWidget(self.createRegionShapeButton("TwoByTwoBox", CDControlPanel.TwoByTwoBoxConst))

       
        self.informationGroupBox.layout().addLayout(sceneItemLayerLayout)


        sceneXSignLabel = QtGui.QLabel()
        sceneXSignLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneXSignLabel.setText("x:")
        sceneXSignLabel.setFont(lFont)
        sceneXSignLabel.setMargin(2)
        sceneYSignLabel = QtGui.QLabel()
        sceneYSignLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneYSignLabel.setText("  y:")
        sceneYSignLabel.setFont(lFont)
        sceneYSignLabel.setMargin(2)
        sceneColorSignLabel = QtGui.QLabel()
        sceneColorSignLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneColorSignLabel.setText("  color:")
        sceneColorSignLabel.setFont(lFont)
        sceneColorSignLabel.setMargin(2)

        self.freehandXLabel = QtGui.QLabel()
        self.freehandXLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.freehandXLabel.setText(" ")
        self.freehandXLabel.setMargin(2)
        self.freehandXLabel.setFont(lFont)
        self.freehandYLabel = QtGui.QLabel()
        self.freehandYLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.freehandYLabel.setText(" ")
        self.freehandYLabel.setMargin(2)
        self.freehandYLabel.setFont(lFont)
        self.freehandColorLabel = QtGui.QLabel()
        self.freehandColorLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.freehandColorLabel.setPixmap(self.createColorPixmap(QtCore.Qt.white))
        self.freehandColorLabel.setMargin(2)
        self.freehandColorLabel.setFont(lFont)

        self.resizingItemLabelWidget = QtGui.QWidget()
        self.resizingItemLabelWidget.setLayout(QtGui.QHBoxLayout())
        self.resizingItemLabelWidget.layout().setMargin(2)
        self.resizingItemLabelWidget.layout().setSpacing(4)
        self.resizingItemLabelWidget.setFont(lFont)
        self.resizingItemLabelWidget.layout().addWidget(sceneXSignLabel)
        self.resizingItemLabelWidget.layout().addWidget(self.freehandXLabel)
        self.resizingItemLabelWidget.layout().addWidget(sceneYSignLabel)
        self.resizingItemLabelWidget.layout().addWidget(self.freehandYLabel)
        self.resizingItemLabelWidget.layout().addWidget(sceneColorSignLabel)
        self.resizingItemLabelWidget.layout().addWidget(self.freehandColorLabel)


        self.informationGroupBox.layout().addWidget(self.resizingItemLabelWidget)           

        self.imageControlsMainLayout.addWidget(self.informationGroupBox)





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
        self.pickingModeGroup.addButton(self.pickExtractCellsRegionButton, CDConstants.ImageModeExtractCells)
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
        if self.pickExtractCellsRegionButton.isChecked():
            lPickingMode = CDConstants.ImageModeExtractCells
        if lPickingMode != self.theInputImagePickingMode:
            self.theInputImagePickingMode = lPickingMode
            # CDConstants.printOut( "the picking mode is ="+str(self.theInputImagePickingMode), CDConstants.DebugTODO )
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
        CDConstants.printOut( "the requested Input Image scale/zoom is = "+str(pValueString), CDConstants.DebugTODO )
        lScaleZoom = pValueString
        if lScaleZoom != self.theScaleZoom:
            self.theScaleZoom = lScaleZoom
            CDConstants.printOut( "the new Input Image scale/zoom will be = "+str(self.theScaleZoom), CDConstants.DebugTODO )
            # propagate the signal upstream, for example to parent objects:
            self.signalImageScaleZoomHasChanged.emit(self.theScaleZoom)



    # ------------------------------------------------------------------
    def setFreehandXLabel(self, pLabelText):
        self.freehandXLabel.setText(pLabelText)
#         CDConstants.printOut( "___ - DEBUG ----- CDControlInputImage: setFreehandXLabel(): done", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setFreehandYLabel(self, pLabelText):
        self.freehandYLabel.setText(pLabelText)
#         CDConstants.printOut( "___ - DEBUG ----- CDControlInputImage: setFreehandYLabel(): done", CDConstants.DebugTODO )


    # ------------------------------------------------------------------
    def setFreehandColorLabel(self, pLabelColor):
        self.freehandColorLabel.setPixmap(self.createColorPixmap(pLabelColor))
#         CDConstants.printOut( "___ - DEBUG ----- CDControlInputImage: setFreehandColorLabel(): done", CDConstants.DebugTODO )

    # ------------------------------------------------------------
    def createColorPixmap(self, color):
        pixmap = QtGui.QPixmap(32, 32)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtCore.Qt.NoPen)
        painter.fillRect(QtCore.QRect(0, 0, 32, 32), QtGui.QBrush(QtGui.QColor(color)))
        painter.end()

        return QtGui.QPixmap(pixmap)


# end class CDControlInputImage(QtGui.QWidget)
# ======================================================================

