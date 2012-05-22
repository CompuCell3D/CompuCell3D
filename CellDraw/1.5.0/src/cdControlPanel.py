#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# ======================================================================
# a QWidget-based control panel, in application-specific panel style
# ======================================================================

class CDControlPanel(QtGui.QWidget):

    # possible types for regions used in the diagram scene:
    RectangleConst, TenByTenBoxConst, StartEndConst, \
        TwoByTwoBoxConst, PathConst = range(5)
   
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def __init__(self, pParent = None):
        super(CDControlPanel,self).__init__(pParent)

        self.parentWindow = pParent
       
        # the miDict dictionary object, local to this class, will contain
        #    references to all global objects declared in the main script,
        #    as set in the finishSetup() method below:
        # self.miDict = dict()


        # a QGroupBox containing buttons for toggling the layer selection
        #    it's assigned below, in setControlsForLayerSelection()
        self.controlsForLayerSelection = 0

        # a QGroupBox with a combobox (pop-up menu) for scene scale/zoom
        #    it's assigned below, in setControlsForSceneScaleZoom()
        self.controlsForSceneScaleZoom = 0

        # a QGroupBox containing buttons for toggling the
        #    drawing mode between regions and cells,
        #    it's assigned below, in setControlsForDrawingRegionOrCellToggle()
        self.controlsForDrawingRegionOrCell = 0

        # a QGroupBox containing buttons for types of regions and cells
        #    it's assigned below, in setControlsForTypes()
        self.controlsForTypes = 0

        # a QGroupBox containing buttons with scene item edit controls,
        #    it's assigned below, in setControlsForSceneItemEdit()
        self.controlsForSceneItemEdit = 0


        # the widgetDict dictionary object is local to this class and
        # contains references to icons:
        self.widgetDict = { \
            CDControlPanel.RectangleConst: QtGui.QIcon(), \
            CDControlPanel.TenByTenBoxConst: QtGui.QIcon(), \
            CDControlPanel.StartEndConst: QtGui.QIcon(), \
            CDControlPanel.TwoByTwoBoxConst: QtGui.QIcon(), \
            CDControlPanel.PathConst: QtGui.QIcon(), \
            }

        # a QButtonGroup containing all buttons for region shapes,
        #    it's assigned below, in setButtonGroupForRegionShapes()
        self.buttonGroupForRegionShapes = 0

        # a QButtonGroup containing all buttons for backgrounds,
        #    it's assigned below, in setButtonGroupForBackgrounds()
        self.buttonGroupForBackgrounds = 0

        # a QGroupBox containing buttons and sliders for controlling the
        #    input image picking mode and rendering,
        #    it's assigned below, in setControlsForInputImagePicking()
        self.controlsForInputImagePicking = 0


        # class globals for the background button from input image:
        # 2010 - Mitja: add code for new backgrounds:
        lBoringPixMap = QtGui.QPixmap(240, 180)
        lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.white) )
        self.imageFromFile = QtGui.QImage(lBoringPixMap)
        self.imageNameFromFile = "BlankBackground"

        # the miImageDocument object holds the image document & pixmap,
        #   the actual object is set in finishSetup() below
        #   (this means that there is no connection to the image file
        #    document until finishSetup() is called) :

        #
        # QWidget setup (1) - windowing GUI setup for Control Panel:
        #

        self.setWindowTitle("Control Panel")
        # QVBoxLayout layout lines up widgets vertically:
        self.mainControlPanelLayout = QtGui.QVBoxLayout()
        self.mainControlPanelLayout.setMargin(2)
        self.mainControlPanelLayout.setSpacing(2)
        self.mainControlPanelLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
#         self.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
#         self.setAutoFillBackground(True)


        self.setLayout(self.mainControlPanelLayout)

        #
        # QWidget setup (2) - more windowing GUI setup for Control Panel:
        #

        miDialogsWindowFlags = QtCore.Qt.WindowFlags()
        # this panel is a so-called "Tool" (by PyQt and Qt definitions)
        #    we'd use the Tool type of window, except for this oh-so typical Qt bug:
        #    http://bugreports.qt.nokia.com/browse/QTBUG-6418
        #    i.e. it defines a system-wide panel which shows on top of *all* applications,
        #    even when this application is in the background.
        # miDialogsWindowFlags = QtCore.Qt.Tool
        #    so we use a plain QtCore.Qt.Window type instead:
        miDialogsWindowFlags = QtCore.Qt.Window
        #    add a peculiar WindowFlags combination to have no close/minimize/maxize buttons:
        miDialogsWindowFlags |= QtCore.Qt.WindowTitleHint
        miDialogsWindowFlags |= QtCore.Qt.CustomizeWindowHint
#        miDialogsWindowFlags |= QtCore.Qt.WindowMinimizeButtonHint
#        miDialogsWindowFlags |= QtCore.Qt.WindowStaysOnTopHint
        self.setWindowFlags(miDialogsWindowFlags)

        # 1. The widget is not modal and does not block input to other widgets.
        # 2. If widget is inactive, the click won't be seen by the widget.
        #    (it does NOT work as Qt docs says it would on Mac OS X: click-throughs don't get disabled)
        # 3. The widget can choose between alternative sizes for widgets to avoid clipping.
        # 4. The native Carbon size grip should be opaque instead of transparent.
        self.setAttribute(QtCore.Qt.NonModal  | \
                          QtCore.Qt.WA_MacNoClickThrough | \
                          QtCore.Qt.WA_MacVariableSize | \
                          QtCore.Qt.WA_MacOpaqueSizeGrip )

        # do not delete the window widget when the window is closed:
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        #
        # QWidget setup - more windowing GUI setup for Control Panel
        #                 continues below, once everything else in
        #                 this application is initialized, in the
        #                 finishSetup() function
        #


        #
        # QWidget setup (3) - more settings for the control panel's GUI:
        #
       
        self.setMinimumSize(256, 624)
        # setGeometry is inherited from QWidget, taking 4 arguments:
        #   x,y  of the top-left corner of the QWidget, from top-left of screen
        #   w,h  of the QWidget
        # self.setGeometry(100,100,300,200)

        # the following is only useful to fix random placement at initialization
        #   *if* we use this panel as stand-alone, without including it in windows etc.
        #   These are X,Y *screen* coordinates (INCLUDING menu bar, etc.),
        #   where X,Y=0,0 is the top-left corner of the screen:
        pos = self.pos()
        pos.setX(10)
        pos.setY(30)
        self.move(pos)
        self.show()
       
        # setWindowOpacity seems to work only if it's set after setting WindowFlags and attributes:
        self.setWindowOpacity(0.95)

        print "___ - DEBUG ----- CDControlPanel: __init__(): done"






    # ------------------------------------------------------------------
    def populateControlPanel(self):

        # ----------------------------------------------------------------
        #
        # QWidget setup (1) - add general controls:
        #

        labelHeaderFont = QtGui.QFont()
        labelHeaderFont.setStyleStrategy(QtGui.QFont.PreferAntialias | QtGui.QFont.PreferQuality)
        labelHeaderFont.setStyleHint(QtGui.QFont.SansSerif)
        labelHeaderFont.setWeight(QtGui.QFont.Bold)

        generalHeaderLabel = QtGui.QLabel("Cell Scene Editing ")
        generalHeaderLabel.setFont(labelHeaderFont)
        generalHeaderLabel.setMargin(0)
        generalHeaderLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.mainControlPanelLayout.addWidget(generalHeaderLabel)


        # ----------------------------------------------------------------
        #
        # QWidget setup (2) - add controls for layer selection, and for
        #       region vs. cell toggle:
        #

        aSimpleQHBoxLayout = QtGui.QHBoxLayout()
        aSimpleQHBoxLayout.setMargin(0)
        aSimpleQHBoxLayout.setSpacing(4)
        aSimpleQHBoxLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
       
        # the layer selection control is defined in its own class:
        aSimpleQHBoxLayout.addWidget(self.controlsForLayerSelection)

        # the "regionOrCell" control is defined in its own class:
        aSimpleQHBoxLayout.addWidget(self.controlsForDrawingRegionOrCell)

        self.mainControlPanelLayout.addLayout(aSimpleQHBoxLayout)

        # ----------------------------------------------------------------
        #
        # QWidget setup (3) - add controls for scene scale/zoom, and for
        #       the types of regions and cells:
        #

        anotherSimpleQHBoxLayout = QtGui.QHBoxLayout()
        anotherSimpleQHBoxLayout.setMargin(0)
        anotherSimpleQHBoxLayout.setSpacing(4)
        anotherSimpleQHBoxLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # the combobox/pop-up menu for scale/zoom control is defined in its own class:
        anotherSimpleQHBoxLayout.addWidget(self.controlsForSceneScaleZoom)

        # the controls for types of regions and cells are defined in its own class:
        anotherSimpleQHBoxLayout.addWidget(self.controlsForTypes)

        self.mainControlPanelLayout.addLayout(anotherSimpleQHBoxLayout)

        self.mainControlPanelLayout.addStretch(10)



        # ----------------------------------------------------------------
        #
        # QWidget setup (4) - add controls for Item editing in the Scene:
        #
        print "___ - DEBUG ----- CDControlPanel: populateControlPanel() 1"

        sceneHeaderLabel = QtGui.QLabel("Cell Scene Layer ")
        sceneHeaderLabel.setFont(labelHeaderFont)
        sceneHeaderLabel.setMargin(0)
        sceneHeaderLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # sceneHeaderLabel.setFrameShape(QtGui.QFrame.Panel)
        # sceneHeaderLabel.setPalette(QtGui.QPalette(QtGui.QColor(QtCore.Qt.lightGray)))
        # sceneHeaderLabel.setAutoFillBackground(True)
        self.mainControlPanelLayout.addWidget(sceneHeaderLabel)





        # ----------------------------------------------------------------
        # add a QGroupBox containing buttons with scene item edit controls,
        #    such as cut/copy/paste/delete etc.
        self.mainControlPanelLayout.addWidget(self.controlsForSceneItemEdit)




        # ----------------------------------------------------------------
        # add a QGroupBox for adding new Items to the Scene:
        #
        self.sceneItemLayerGroupBox = QtGui.QGroupBox("New Item")
        #         self.sceneItemLayerGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
        #         self.sceneItemLayerGroupBox.setAutoFillBackground(True)
        self.sceneItemLayerGroupBox.setLayout(QtGui.QVBoxLayout())
        self.sceneItemLayerGroupBox.layout().setMargin(2)
        self.sceneItemLayerGroupBox.layout().setSpacing(4)
        self.sceneItemLayerGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)


        lFont = QtGui.QFont()
        lFont.setWeight(QtGui.QFont.Light)

        sceneItemLayerLayout = QtGui.QHBoxLayout()
        sceneItemLayerLayout.setMargin(0)
        sceneItemLayerLayout.setSpacing(4)
        sceneItemLayerLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        print "___ - DEBUG ----- CDControlPanel: 1"
        sceneItemLayerLayout.addWidget(self.createRegionShapeButton("Ellipse", CDControlPanel.PathConst))

        print "___ - DEBUG ----- CDControlPanel: 2"
        sceneItemLayerLayout.addWidget(self.createRegionShapeButton("Rectangle", CDControlPanel.RectangleConst))

        print "___ - DEBUG ----- CDControlPanel: empty"
        sceneItemLayerLayout.addSpacing(40)

        print "___ - DEBUG ----- CDControlPanel: 3"
        sceneItemLayerLayout.addWidget(self.createRegionShapeButton("TenByTenBox", CDControlPanel.TenByTenBoxConst))

        print "___ - DEBUG ----- CDControlPanel: 4"
        sceneItemLayerLayout.addWidget(self.createRegionShapeButton("TwoByTwoBox", CDControlPanel.TwoByTwoBoxConst))

       
        self.sceneItemLayerGroupBox.layout().addLayout(sceneItemLayerLayout)


        sceneXSignLabel = QtGui.QLabel()
        sceneXSignLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneXSignLabel.setText("x:")
        sceneXSignLabel.setFont(lFont)
        sceneXSignLabel.setMargin(0)
        sceneYSignLabel = QtGui.QLabel()
        sceneYSignLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneYSignLabel.setText("  y:")
        sceneYSignLabel.setFont(lFont)
        sceneYSignLabel.setMargin(0)
        sceneWidthSignLabel = QtGui.QLabel()
        sceneWidthSignLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneWidthSignLabel.setText("  w:")
        sceneWidthSignLabel.setMargin(0)
        sceneWidthSignLabel.setFont(lFont)
        sceneHeightSignLabel = QtGui.QLabel()
        sceneHeightSignLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneHeightSignLabel.setText("  h:")
        sceneHeightSignLabel.setMargin(0)
        sceneHeightSignLabel.setFont(lFont)

        self.resizingItemXLabel = QtGui.QLabel()
        self.resizingItemXLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.resizingItemXLabel.setText(" ")
        self.resizingItemXLabel.setMargin(0)
        self.resizingItemXLabel.setFont(lFont)
        self.resizingItemYLabel = QtGui.QLabel()
        self.resizingItemYLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.resizingItemYLabel.setText(" ")
        self.resizingItemYLabel.setMargin(0)
        self.resizingItemYLabel.setFont(lFont)
        self.resizingItemWidthLabel = QtGui.QLabel()
        self.resizingItemWidthLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.resizingItemWidthLabel.setText(" ")
        self.resizingItemWidthLabel.setMargin(0)
        self.resizingItemWidthLabel.setFont(lFont)

        self.resizingItemHeightLabel = QtGui.QLabel()
        self.resizingItemHeightLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.resizingItemHeightLabel.setText(" ")
        self.resizingItemHeightLabel.setMargin(0)
        self.resizingItemHeightLabel.setFont(lFont)

        self.resizingItemLabelWidget = QtGui.QWidget()
        self.resizingItemLabelWidget.setLayout(QtGui.QHBoxLayout())
        self.resizingItemLabelWidget.layout().setMargin(0)
        self.resizingItemLabelWidget.layout().setSpacing(0)
        self.resizingItemLabelWidget.setFont(lFont)
        self.resizingItemLabelWidget.layout().addWidget(sceneXSignLabel)
        self.resizingItemLabelWidget.layout().addWidget(self.resizingItemXLabel)
        self.resizingItemLabelWidget.layout().addWidget(sceneYSignLabel)
        self.resizingItemLabelWidget.layout().addWidget(self.resizingItemYLabel)
        self.resizingItemLabelWidget.layout().addWidget(sceneWidthSignLabel)
        self.resizingItemLabelWidget.layout().addWidget(self.resizingItemWidthLabel)
        self.resizingItemLabelWidget.layout().addWidget(sceneHeightSignLabel)
        self.resizingItemLabelWidget.layout().addWidget(self.resizingItemHeightLabel)


        self.sceneItemLayerGroupBox.layout().addWidget(self.resizingItemLabelWidget)           

        self.mainControlPanelLayout.addWidget(self.sceneItemLayerGroupBox)


        # ----------------------------------------------------------------
        print "___ - DEBUG ----- CDControlPanel: populateControlPanel() 3"

        self.sceneBackgroundLayerGroupBox = QtGui.QGroupBox("Background")
#         self.sceneBackgroundLayerGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
#         self.sceneBackgroundLayerGroupBox.setAutoFillBackground(True)
        self.sceneBackgroundLayerGroupBox.setLayout(QtGui.QVBoxLayout())
        self.sceneBackgroundLayerGroupBox.layout().setMargin(2)
        self.sceneBackgroundLayerGroupBox.layout().setSpacing(4)
        self.sceneBackgroundLayerGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.backgroundLayout = QtGui.QGridLayout()
        self.backgroundLayout.setMargin(0)
        self.backgroundLayout.setSpacing(4)
        self.backgroundLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.theBackgroundImageCellWidget = self.createBackgroundCellWidgetFromImage( \
                self.imageNameFromFile, self.imageFromFile)

        self.backgroundLayout.addWidget(self.theBackgroundImageCellWidget, 0, 0)

        self.backgroundLayout.addWidget(self.createBackgroundCellWidget("No Grid",
                ':/icons/background4.png'), 0, 1)
        self.backgroundLayout.addWidget(self.createBackgroundCellWidget("White Grid",
                ':/icons/background2.png'), 0, 2)
        self.backgroundLayout.addWidget(self.createBackgroundCellWidget("Gray Grid",
                ':/icons/background3.png'), 0, 3)
        self.backgroundLayout.addWidget(self.createBackgroundCellWidget("Blue Grid",
                ':/icons/background1.png'), 0, 4)

        # 2011 - Mitja: make sure that one background button is checked:
        theBgButtons = self.buttonGroupForBackgrounds.buttons()
        for myButton in theBgButtons:
            print "123123123123123 =", myButton, myButton.text()
            if myButton.text() == "No Grid":
                myButton.setChecked(True)

#         self.backgroundLayout.setRowStretch(2, 10)
#         self.backgroundLayout.setColumnStretch(2, 10)

        self.sceneBackgroundLayerGroupBox.layout().addLayout(self.backgroundLayout)

        self.mainControlPanelLayout.addWidget(self.sceneBackgroundLayerGroupBox)





        # ----------------------------------------------------------------
        print "___ - DEBUG ----- CDControlPanel: populateControlPanel() 4"
       

        self.pifDimensionsGroupBox = QtGui.QGroupBox("Scene Size")
#         self.pifDimensionsGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
#         self.pifDimensionsGroupBox.setAutoFillBackground(True)
        self.pifDimensionsGroupBox.setLayout(QtGui.QHBoxLayout())
        self.pifDimensionsGroupBox.layout().setMargin(2)
        self.pifDimensionsGroupBox.layout().setSpacing(4)
        self.pifDimensionsGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # 2010 - Mitja: add a widget displaying the scene dimensions at all times:
        # the scene dimension widget will have a title label:
        self.sceneWidthLabel = QtGui.QLabel()
        self.sceneWidthLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.sceneWidthLabel.setText("w")
        self.sceneWidthLabel.setFont(lFont)
        self.sceneWidthLabel.setMargin(0)
        # self.sceneWidthLabel.setFrameStyle(QtGui.QFrame.StyledPanel | QtGui.QFrame.Sunken)
        sceneTimesSignLabel = QtGui.QLabel()
        sceneTimesSignLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneTimesSignLabel.setText( u"\u00D7" ) # <-- the multiplication sign as unicode
        sceneTimesSignLabel.setFont(lFont)
        sceneTimesSignLabel.setMargin(0)
        # sceneTimesSignLabel.setFrameStyle(QtGui.QFrame.StyledPanel | QtGui.QFrame.Sunken)
        self.sceneHeightLabel = QtGui.QLabel()
        self.sceneHeightLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.sceneHeightLabel.setText("  h:")
        self.sceneHeightLabel.setFont(lFont)
        self.sceneHeightLabel.setMargin(0)
        # self.sceneHeightLabel.setFrameStyle(QtGui.QFrame.StyledPanel | QtGui.QFrame.Sunken)
        self.sceneUnitsLabel = QtGui.QLabel()
        self.sceneUnitsLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.sceneUnitsLabel.setText("  units:")
        self.sceneUnitsLabel.setFont(lFont)
        self.sceneUnitsLabel.setMargin(0)

#         self.sceneDimensionsWidget = QtGui.QWidget()
#         # self.sceneDimensionsWidget.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
#         # self.sceneDimensionsWidget.setAutoFillBackground(True)
#         self.sceneDimensionsWidget.setLayout(QtGui.QHBoxLayout())
#         self.sceneDimensionsWidget.layout().setMargin(0)
#         self.sceneDimensionsWidget.layout().setSpacing(2)
#         self.sceneDimensionsWidget.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
#         self.sceneDimensionsWidget.setFont(lFont)

        self.sceneDimensionsWidget = QtGui.QWidget()
        self.sceneDimensionsWidget.setLayout(QtGui.QHBoxLayout())
        self.sceneDimensionsWidget.layout().setMargin(0)
        self.sceneDimensionsWidget.layout().setSpacing(8)
        self.sceneDimensionsWidget.setFont(lFont)
        self.sceneDimensionsWidget.layout().addWidget(self.sceneWidthLabel)
        self.sceneDimensionsWidget.layout().addWidget(sceneTimesSignLabel)
        self.sceneDimensionsWidget.layout().addWidget(self.sceneHeightLabel)
        self.sceneDimensionsWidget.layout().addWidget(self.sceneUnitsLabel)

        self.pifDimensionsGroupBox.layout().addWidget(self.sceneDimensionsWidget)

        self.mainControlPanelLayout.addWidget(self.pifDimensionsGroupBox)







        self.mainControlPanelLayout.addStretch(10)

        # ----------------------------------------------------------------
        #
        # QWidget setup (4) - add file labels and controls for Image Layer content:
        #
        imageHeaderLabel = QtGui.QLabel("Image Layer")
        imageHeaderLabel.setFont(labelHeaderFont)
        sceneHeaderLabel.setMargin(0)
        sceneHeaderLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.mainControlPanelLayout.addWidget(imageHeaderLabel)

        # this QGroupBox is defined in its own class:
        self.mainControlPanelLayout.addWidget(self.controlsForInputImagePicking)


#         self.pifWidthXLabel = QtGui.QLabel("PIFF grid width: x=")
#         self.pifWidthXInputBox = QtGui.QSpinBox()
#         self.pifWidthXInputBox.setRange(0, 2000)
#         self.mainControlPanelLayout.addWidget(self.pifWidthXLabel)
#         self.mainControlPanelLayout.addWidget(self.pifWidthXInputBox)
#
#         self.pifHeightYLabel = QtGui.QLabel("PIFF grid height: y=")
#         self.pifHeightYInputBox = QtGui.QSpinBox()
#         self.pifHeightYInputBox.setRange(0, 2000)
#         self.mainControlPanelLayout.addWidget(self.pifHeightYLabel)
#         self.mainControlPanelLayout.addWidget(self.pifHeightYInputBox)

        # self.mainControlPanelLayout.addSeparator()
        self.mainControlPanelLayout.addStretch(10)
# NOW ADD A GUI for the grid and the controls here,
# and a GUI for the image controls on the toolbar on the window!
        print "___ - DEBUG ----- CDControlPanel: finishSetup(): done"





    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def miCreateSlider(self, pChangedSignal, pSetterSlot):
        lSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        lSlider.setRange(0, 360 * 16)
        lSlider.setSingleStep(16)
        lSlider.setPageStep(15 * 16)
        lSlider.setTickInterval(15 * 16)
        lSlider.setTickPosition(QtGui.QSlider.TicksRight)
        # events (signals/slots) :
        lSlider.valueChanged.connect(pSetterSlot)
        pChangedSignal.connect(lSlider.setValue)
        # return the QSlider that's just been prepared :
        return lSlider
        print "___ - DEBUG ----- CDControlPanel: miCreateSlider(): done"


    # ------------------------------------------------------------
    def createRegionShapeButton(self, pText, pDiagramType):

        # 2010 - Mitja: add code for handling insertion of path-derived items:
        if (pDiagramType == CDControlPanel.PathConst) :
            # we are instantiating a diagram item derived from a path:
            # 2010 - Mitja: if there is no Path yet, make it a simple boring one:
            miBoringPath = QtGui.QPainterPath()
            miBoringPath.addEllipse(-100.0, -50.0, 200.0, 100.0)
            polygo = miBoringPath.toFillPolygon()

            pixmap = QtGui.QPixmap(210, 210)
            pixmap.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(pixmap)
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 8.0))
            painter.translate(105, 105)

            #  self.setFillRule(QtCore.Qt.WindingFill) from Qt documentation:
            # Specifies that the region is filled using the non zero winding rule.
            # With this rule, we determine whether a point is inside the shape by
            # using the following method. Draw a horizontal line from the point to a
            # location outside the shape. Determine whether the direction of the line
            # at each intersection point is up or down. The winding number is
            # determined by summing the direction of each intersection. If the number
            # is non zero, the point is inside the shape. This fill mode can also in
            # most cases be considered as the intersection of closed shapes.
            painter.drawPolygon(polygo, QtCore.Qt.WindingFill)

            lIcon = QtGui.QIcon(pixmap)
            painter.end()

        else :
            # we are instantiating a diagram item for a normal polygon type:
            lIcon = self.widgetDict[pDiagramType]

        # create an icon:
        icon = QtGui.QIcon(lIcon)

        button = QtGui.QToolButton()
        button.setIcon(icon)
        button.setIconSize(QtCore.QSize(24, 24))
        button.setCheckable(True)

        self.buttonGroupForRegionShapes.addButton(button, pDiagramType)

        layout = QtGui.QGridLayout()
        layout.setMargin(0)
        layout.addWidget(button, 0, 0, QtCore.Qt.AlignHCenter)
        # layout.addWidget(QtGui.QLabel(pText), 1, 0, QtCore.Qt.AlignCenter)

        widget = QtGui.QWidget()
        widget.setLayout(layout)

        return widget
        # return button
        print "___ - DEBUG ----- CDControlPanel: createRegionShapeButton(): done"



    # ------------------------------------------------------------
    def createBackgroundCellWidget(self, text, pPixmap):
        button = QtGui.QToolButton()
        button.setText(text)
        button.setIcon(QtGui.QIcon(pPixmap))
        button.setIconSize(QtCore.QSize(24, 24))
        button.setCheckable(True)
        self.buttonGroupForBackgrounds.addButton(button)

        layout = QtGui.QGridLayout()
        layout.setMargin(0)
        layout.addWidget(button, 0, 0, QtCore.Qt.AlignHCenter)
        # layout.addWidget(QtGui.QLabel(text), 1, 0, QtCore.Qt.AlignCenter)

        widget = QtGui.QWidget()
        widget.setLayout(layout)

        return widget


    # ------------------------------------------------------------
    # 2010 - Mitja: add code for new backgrounds:
    # ------------------------------------------------------------
    def createBackgroundCellWidgetFromImage(self, text, pImage):
        button = QtGui.QToolButton()
        button.setText(text)
        # treat the "image" parameter actually as an image, and not as a pixmap:
        button.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(pImage)))
        button.setIconSize(QtCore.QSize(24, 24))
        button.setCheckable(True)
        self.buttonGroupForBackgrounds.addButton(button)

        layout = QtGui.QGridLayout()
        layout.setMargin(0)
        layout.addWidget(button, 0, 0, QtCore.Qt.AlignHCenter)
        # layout.addWidget(QtGui.QLabel(text), 1, 0, QtCore.Qt.AlignCenter)

        widget = QtGui.QWidget()
        widget.setLayout(layout)

        return widget




    # ------------------------------------------------------------
    # 2010 - Mitja: add code for new backgrounds:
    # ------------------------------------------------------------
    def updateBackgroundImageButtons(self, pText, pImage):
        # store parameters into globals:
        self.imageFromFile = pImage
        self.imageNameFromFile = pText

        # select the 2nd toolbox item, i.e. the "Backgrounds":
        # self.toolBox.setCurrentIndex(1)
        # remove the first widget in the layout:
        self.backgroundLayout.removeWidget(self.theBackgroundImageCellWidget)
        # the above "removeWidget" statement does not seem to be enough. One must hide the widget too:
        self.theBackgroundImageCellWidget.hide()

        # create a new background cell widget:
        self.theBackgroundImageCellWidget = None # <----- do we need this ???
        self.theBackgroundImageCellWidget = self.createBackgroundCellWidgetFromImage(pText, self.imageFromFile)
        # add the new background cell widget to the layout:
        # self.toolBox.currentWidget.layout.addWidget(self.theBackgroundImageCellWidget, 0, 0)
        self.backgroundLayout.addWidget(self.theBackgroundImageCellWidget, 0, 0)

        # this would update the scene background image to the new bacgkround image,
        #   regardless of what is currently selected as background. that's not what we want:
        #   users ought to be able to keep the background they had selected before.
        # self.updateSceneBackgroundImage(self.imageNameFromFile)
       
        # if the currently selected background is the one from image
        #     then call updateSceneBackgroundImage since it needs to show the new image
        #     otherwise leave background image as is (e.g. white, etc)
        if self.isTheBackgroundAnImage() is True:
            if type(self.parentWindow).__name__ == "CDDiagramSceneMainWidget":
                self.parentWindow.updateSceneBackgroundImage(self.imageNameFromFile)
                print "self.parentWindow.updateSceneBackgroundImage(self.imageNameFromFile) DONE"

        # 2011 - Mitja: also update the button for selecting the Image Layer in the editor:
        self.controlsForLayerSelection.setImageLayerButtonIcon( QtGui.QIcon( QtGui.QPixmap.fromImage(pImage) ) )


        print "___ - DEBUG ----- CDControlPanel: updateBackgroundImageButtons() done."



    # ------------------------------------------------------------
    # 2010 - Mitja - add a fuction to quickly find out whether the scene background
    #     is set to a user-loaded image or to one of the plain built-in patterns:
    # ------------------------------------------------------------
    def isTheBackgroundAnImage(self):
        buttons = self.buttonGroupForBackgrounds.buttons()
        for myButton in buttons:
            print "123123123123123 =", myButton, myButton.text()       
            if myButton.isChecked() is True:
                lText = myButton.text()

        if (lText == "Blue Grid") or (lText == "White Grid") or (lText == "Gray Grid") or (lText == "No Grid"):
            return False
        else:
            return True

        print "___ - DEBUG ----- CDControlPanel: isTheBackgroundAnImage() done."



    # ------------------------------------------------------------
    def setWidgetIcon(self, pDiagramType, pIcon):
        self.widgetDict[pDiagramType] = pIcon
        print "___ - DEBUG ----- CDControlPanel: setWidgetIcon(): done"



    # ------------------------------------------------------------
    def setControlsForDrawingRegionOrCellToggle(self, pWidget):
        self.controlsForDrawingRegionOrCell = pWidget
        print "___ - DEBUG ----- CDControlPanel: setControlsForDrawingRegionOrCellToggle(): done"


    # ------------------------------------------------------------
    def setControlsForTypes(self, pWidget):
        self.controlsForTypes = pWidget
        print "___ - DEBUG ----- CDControlPanel: setControlsForTypes(): done"


    # ------------------------------------------------------------
    def setControlsForLayerSelection(self, pWidget):
        self.controlsForLayerSelection = pWidget
        print "___ - DEBUG ----- CDControlPanel: setControlsForLayerSelection(): done"

    # ------------------------------------------------------------
    def setControlsForSceneScaleZoom(self, pWidget):
        self.controlsForSceneScaleZoom = pWidget
        print "___ - DEBUG ----- CDControlPanel: setControlsForSceneScaleZoom(): done"


    # ------------------------------------------------------------
    def setControlsForInputImagePicking(self, pGroupBox):
        self.controlsForInputImagePicking = pGroupBox
        print "___ - DEBUG ----- CDControlPanel: setControlsForInputImagePicking(): done"


    # ------------------------------------------------------------
    def setControlsForSceneItemEdit(self, pGroupBox):
        self.controlsForSceneItemEdit = pGroupBox
        print "___ - DEBUG ----- CDControlPanel: setControlsForSceneItemEdit(): done"








    # ------------------------------------------------------------
    def setButtonGroupForRegionShapes(self, pButtonGroup):
        self.buttonGroupForRegionShapes = pButtonGroup
        print "___ - DEBUG ----- CDControlPanel: setButtonGroupForRegionShapes(): done"


    # ------------------------------------------------------------
    def setButtonGroupForBackgrounds(self, pButtonGroup):
        self.buttonGroupForBackgrounds = pButtonGroup
        print "___ - DEBUG ----- CDControlPanel: setButtonGroupForRegionShapes(): done"




    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def setImageFileNameLabel(self, pImageFileName):
        # save the image file name into the label to show it in the panel
        self.imageFileNameLabel.setText(pImageFileName)
        print "___ - DEBUG ----- CDControlPanel: setImageFileNameLabel(): done"


    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def setPiffFileNameLabel(self, pPifFileName):
        # save the PIFF file name into the label to show it in the panel
        self.piffFileNameLabel.setText(pPifFileName)
        print "___ - DEBUG ----- CDControlPanel: setPiffFileNameLabel(): done"

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
#     def setWindowFlags(self, pFlags):
#         super(CDControlPanel, self).setWindowFlags(pFlags)
#         print "___ - DEBUG ----- CDControlPanel: setWindowFlags(): done"






    # ------------------------------------------------------------------
    # functions for externally setting the control panel's label values:

    # ------------------------------------------------------------------
    def setSceneWidthLabel(self, pSceneWidthLabel):
        self.sceneWidthLabel.setText(pSceneWidthLabel)
        print "___ - DEBUG ----- CDControlPanel: setSceneWidthLabel(): done"

    # ------------------------------------------------------------------
    def setSceneHeightLabel(self, pSceneHeightLabel):
        self.sceneHeightLabel.setText(pSceneHeightLabel)
        print "___ - DEBUG ----- CDControlPanel: setSceneHeightLabel(): done"

    # ------------------------------------------------------------------
    def setSceneUnitsLabel(self, pSceneUnitsLabel):
        self.sceneUnitsLabel.setText(pSceneUnitsLabel)
        print "___ - DEBUG ----- CDControlPanel: setSceneUnitsLabel(): done"


    # ------------------------------------------------------------------
    def setResizingItemXLabel(self, pLabelText):
        self.resizingItemXLabel.setText(pLabelText)
        print "___ - DEBUG ----- CDControlPanel: setResizingItemXLabel(): done"

    # ------------------------------------------------------------------
    def setResizingItemYLabel(self, pLabelText):
        self.resizingItemYLabel.setText(pLabelText)
        print "___ - DEBUG ----- CDControlPanel: setResizingItemYLabel(): done"

    # ------------------------------------------------------------------
    def setResizingItemWidthLabel(self, pLabelText):
        self.resizingItemWidthLabel.setText(pLabelText)
        print "___ - DEBUG ----- CDControlPanel: setResizingItemWidthLabel(): done"

    # ------------------------------------------------------------------
    def setResizingItemHeightLabel(self, pLabelText):
        self.resizingItemHeightLabel.setText(pLabelText)
        print "___ - DEBUG ----- CDControlPanel: setResizingItemHeightLabel(): done"




# end class CDControlPanel(QtGui.QWidget)
# ======================================================================

# Local Variables:
# coding: US-ASCII
# End:
