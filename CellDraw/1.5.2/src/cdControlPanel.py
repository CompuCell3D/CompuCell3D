#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants


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

        # a QGroupBox containing buttons and sliders for controlling the
        #    sequence of images loaded into a stack,
        #    it's assigned below, in setcontrolsForImageSequence()
        self.controlsForImageSequence = 0


        # a QGroupBox containing buttons and sliders for controlling the
        #    sequence of images loaded into a stack,
        #    it's assigned below, in setControlsForClusters()
        self.controlsForClusters = 0


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
        self.mainControlPanelLayout.setMargin(8)
        self.mainControlPanelLayout.setSpacing(4)
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
       
        self.setMinimumSize(302, 604)
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

        CDConstants.printOut( "___ - DEBUG ----- CDControlPanel: __init__(): done" , CDConstants.DebugTODO )






    # ------------------------------------------------------------------
    def populateControlPanel(self):

        # some commonly used parameters:
        #
        labelHeaderFont = QtGui.QFont()
        labelHeaderFont.setStyleStrategy(QtGui.QFont.PreferAntialias | QtGui.QFont.PreferQuality)
        labelHeaderFont.setStyleHint(QtGui.QFont.SansSerif)
        labelHeaderFont.setWeight(QtGui.QFont.Bold)


# ----------------------------------------------------------------
#
# QWidget setup (0) - add a QTreeView:
# 
#         lTheMainTreeWidget  = QtGui.QTreeWidget( self )
#         lTheMainTreeWidget.setHeaderLabels(QtCore.QString("The QTree"))
#         lTheMainTreeWidget.setColumnCount(1)
#         lTheMainTreeWidget.clear()
#         
#         testHeaderLabel = QtGui.QLabel("Cell Scene Editing ")
#         testHeaderLabel.setFont(labelHeaderFont)
#         testHeaderLabel.setMargin(2)
#         testHeaderLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
# 
#         lTestTreeWidgetItem = QtGui.QTreeWidgetItem(lTheMainTreeWidget)
#         lTheMainTreeWidget.setItemWidget(lTestTreeWidgetItem, 0, testHeaderLabel)
# 
#         self.mainControlPanelLayout.addWidget(lTheMainTreeWidget)
#


        # ----------------------------------------------------------------
        #
        # QWidget setup (1a) - add general controls:
        #

#         generalHeaderLabel = QtGui.QLabel("Cell Scene Editing ")
#         generalHeaderLabel.setFont(labelHeaderFont)
#         generalHeaderLabel.setMargin(2)
#         generalHeaderLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
# 
#         self.mainControlPanelLayout.addWidget(generalHeaderLabel)





        # ----------------------------------------------------------------
        #
        # QWidget setup (2) - prepare for a QTabView for Scene Layer controls:
        #
        #
        lTheCellSceneLayerTab = QtGui.QWidget()
        lTheCellSceneLayerTabLayout = QtGui.QVBoxLayout()
        lTheCellSceneLayerTabLayout.setMargin(2)
        lTheCellSceneLayerTabLayout.setSpacing(4)
        lTheCellSceneLayerTabLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        lTheCellSceneLayerTab.setLayout(lTheCellSceneLayerTabLayout)
        

        sceneHeaderLabel = QtGui.QLabel("Cell Scene Layer ")
        sceneHeaderLabel.setFont(labelHeaderFont)
        sceneHeaderLabel.setMargin(2)
        sceneHeaderLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # sceneHeaderLabel.setFrameShape(QtGui.QFrame.Panel)
        # sceneHeaderLabel.setPalette(QtGui.QPalette(QtGui.QColor(QtCore.Qt.lightGray)))
        # sceneHeaderLabel.setAutoFillBackground(True)
        lTheCellSceneLayerTabLayout.addWidget(sceneHeaderLabel)



        # ----------------------------------------------------------------
        #
        # QWidget setup (2a) - add controls for scene editing mode (pick/move vs. scale)
        #    and for scene zoom:
        #
        lFirstSimpleQHBoxLayout = QtGui.QHBoxLayout()
        lFirstSimpleQHBoxLayout.setMargin(2)
        lFirstSimpleQHBoxLayout.setSpacing(4)
        lFirstSimpleQHBoxLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # the layer selection control is defined in its own class:
        #
        lFirstSimpleQHBoxLayout.addWidget(self.controlsForLayerSelection)

        # the combobox/pop-up menu for scale/zoom control is defined in its own class:
        #
        lFirstSimpleQHBoxLayout.addWidget(self.controlsForSceneScaleZoom)

        lTheCellSceneLayerTabLayout.addLayout(lFirstSimpleQHBoxLayout)







        # ----------------------------------------------------------------
        #
        # QWidget setup (2a) - add controls for Item editing in the Scene:
        #
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: populateControlPanel() 1" )+" ", CDConstants.DebugTODO )
        #



        # ----------------------------------------------------------------
        #
        # QWidget setup (2a) - add a QGroupBox containing buttons with
        #    scene item edit controls, such as cut/copy/paste/delete etc.
        #
        # this is the "Item Edit" QGroupBox, defined in cdControlLayerSelection.py :
        #
        lTheCellSceneLayerTabLayout.addWidget(self.controlsForSceneItemEdit)




        # ----------------------------------------------------------------
        #
        # QWidget setup (2b) - add a QGroupBox for adding new Items to the Scene:
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
        sceneItemLayerLayout.setMargin(2)
        sceneItemLayerLayout.setSpacing(4)
        sceneItemLayerLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: 1" )+" ", CDConstants.DebugTODO )
        sceneItemLayerLayout.addWidget(self.createRegionShapeButton("Ellipse", CDControlPanel.PathConst))

        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: 2" )+" ", CDConstants.DebugTODO )
        sceneItemLayerLayout.addWidget(self.createRegionShapeButton("Rectangle", CDControlPanel.RectangleConst))

        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: empty" )+" ", CDConstants.DebugTODO )
        sceneItemLayerLayout.addSpacing(40)

        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: 3" )+" ", CDConstants.DebugTODO )
        sceneItemLayerLayout.addWidget(self.createRegionShapeButton("TenByTenBox", CDControlPanel.TenByTenBoxConst))

        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: 4" )+" ", CDConstants.DebugTODO )
        sceneItemLayerLayout.addWidget(self.createRegionShapeButton("TwoByTwoBox", CDControlPanel.TwoByTwoBoxConst))

       
        self.sceneItemLayerGroupBox.layout().addLayout(sceneItemLayerLayout)


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
        sceneWidthSignLabel = QtGui.QLabel()
        sceneWidthSignLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneWidthSignLabel.setText("  w:")
        sceneWidthSignLabel.setMargin(2)
        sceneWidthSignLabel.setFont(lFont)
        sceneHeightSignLabel = QtGui.QLabel()
        sceneHeightSignLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneHeightSignLabel.setText("  h:")
        sceneHeightSignLabel.setMargin(2)
        sceneHeightSignLabel.setFont(lFont)

        self.resizingItemXLabel = QtGui.QLabel()
        self.resizingItemXLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.resizingItemXLabel.setText(" ")
        self.resizingItemXLabel.setMargin(2)
        self.resizingItemXLabel.setFont(lFont)
        self.resizingItemYLabel = QtGui.QLabel()
        self.resizingItemYLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.resizingItemYLabel.setText(" ")
        self.resizingItemYLabel.setMargin(2)
        self.resizingItemYLabel.setFont(lFont)
        self.resizingItemWidthLabel = QtGui.QLabel()
        self.resizingItemWidthLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.resizingItemWidthLabel.setText(" ")
        self.resizingItemWidthLabel.setMargin(2)
        self.resizingItemWidthLabel.setFont(lFont)
        self.resizingItemHeightLabel = QtGui.QLabel()
        self.resizingItemHeightLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.resizingItemHeightLabel.setText(" ")
        self.resizingItemHeightLabel.setMargin(2)
        self.resizingItemHeightLabel.setFont(lFont)

        self.resizingItemLabelWidget = QtGui.QWidget()
        self.resizingItemLabelWidget.setLayout(QtGui.QHBoxLayout())
        self.resizingItemLabelWidget.layout().setMargin(2)
        self.resizingItemLabelWidget.layout().setSpacing(4)
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

        lTheCellSceneLayerTabLayout.addWidget(self.sceneItemLayerGroupBox)



        # ----------------------------------------------------------------
        #
        # QWidget setup (2c) - add controls for item types:
        #
        anotherSimpleQHBoxLayout = QtGui.QHBoxLayout()
        anotherSimpleQHBoxLayout.setMargin(2)
        anotherSimpleQHBoxLayout.setSpacing(4)
        anotherSimpleQHBoxLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # the "regionOrCell" control is defined in its own class:
        #
        anotherSimpleQHBoxLayout.addWidget(self.controlsForDrawingRegionOrCell)

        # the controls for types of regions and cells are defined in its own class:
        #
        anotherSimpleQHBoxLayout.addWidget(self.controlsForTypes)

        # add both these two controls as hbox to the scene tab's layout:
        lTheCellSceneLayerTabLayout.addLayout(anotherSimpleQHBoxLayout)


        # ----------------------------------------------------------------
        #
        # QWidget setup (2d) - add a QGroupBox for selecting Scene backrounds:
        #
        self.sceneBackgroundLayerGroupBox = QtGui.QGroupBox("Background")
#         self.sceneBackgroundLayerGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
#         self.sceneBackgroundLayerGroupBox.setAutoFillBackground(True)
        self.sceneBackgroundLayerGroupBox.setLayout(QtGui.QVBoxLayout())
        self.sceneBackgroundLayerGroupBox.layout().setMargin(2)
        self.sceneBackgroundLayerGroupBox.layout().setSpacing(4)
        self.sceneBackgroundLayerGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.backgroundLayout = QtGui.QGridLayout()
        self.backgroundLayout.setMargin(2)
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
            CDConstants.printOut( "123123123123123    myButton="+str(myButton)+"myButton.text()="+str(myButton.text()), CDConstants.DebugTODO )
            if myButton.text() == "No Grid":
                myButton.setChecked(True)

#         self.backgroundLayout.setRowStretch(2, 10)
#         self.backgroundLayout.setColumnStretch(2, 10)

        self.sceneBackgroundLayerGroupBox.layout().addLayout(self.backgroundLayout)

        lTheCellSceneLayerTabLayout.addWidget(self.sceneBackgroundLayerGroupBox)


        # ----------------------------------------------------------------
        #
        # QWidget setup (2e) - add a QGroupBox for showing Scene dimensions:
        #
        self.pifDimensionsGroupBox = QtGui.QGroupBox("Scene Dimensions")
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
        self.sceneWidthLabel.setMargin(2)
        sceneTimesSignLabel = QtGui.QLabel()
        sceneTimesSignLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneTimesSignLabel.setText( u"\u00D7" ) # <-- the multiplication sign as unicode
        sceneTimesSignLabel.setFont(lFont)
        sceneTimesSignLabel.setMargin(2)
        self.sceneHeightLabel = QtGui.QLabel()
        self.sceneHeightLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.sceneHeightLabel.setText("h")
        self.sceneHeightLabel.setFont(lFont)
        self.sceneHeightLabel.setMargin(2)
        sceneTimesSign2Label = QtGui.QLabel()
        sceneTimesSign2Label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        sceneTimesSign2Label.setText( u"\u00D7" ) # <-- the multiplication sign as unicode
        sceneTimesSign2Label.setFont(lFont)
        sceneTimesSign2Label.setMargin(2)
        self.sceneDepthLabel = QtGui.QLabel()
        self.sceneDepthLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.sceneDepthLabel.setText("d")
        self.sceneDepthLabel.setFont(lFont)
        self.sceneDepthLabel.setMargin(2)
        self.sceneUnitsLabel = QtGui.QLabel()
        self.sceneUnitsLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.sceneUnitsLabel.setText("  units:")
        self.sceneUnitsLabel.setFont(lFont)
        self.sceneUnitsLabel.setMargin(2)

#         self.sceneDimensionsWidget = QtGui.QWidget()
#         # self.sceneDimensionsWidget.setPalette(QtGui.QPalette(QtGui.QColor(222,222,222)))
#         # self.sceneDimensionsWidget.setAutoFillBackground(True)
#         self.sceneDimensionsWidget.setLayout(QtGui.QHBoxLayout())
#         self.sceneDimensionsWidget.layout().setMargin(2)
#         self.sceneDimensionsWidget.layout().setSpacing(4)
#         self.sceneDimensionsWidget.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
#         self.sceneDimensionsWidget.setFont(lFont)

        self.sceneDimensionsWidget = QtGui.QWidget()
        self.sceneDimensionsWidget.setLayout(QtGui.QHBoxLayout())
        self.sceneDimensionsWidget.layout().setMargin(2)
        self.sceneDimensionsWidget.layout().setSpacing(4)
        self.sceneDimensionsWidget.setFont(lFont)
        self.sceneDimensionsWidget.layout().addWidget(self.sceneWidthLabel)
        self.sceneDimensionsWidget.layout().addWidget(sceneTimesSignLabel)
        self.sceneDimensionsWidget.layout().addWidget(self.sceneHeightLabel)
        self.sceneDimensionsWidget.layout().addWidget(sceneTimesSign2Label)
        self.sceneDimensionsWidget.layout().addWidget(self.sceneDepthLabel)
        self.sceneDimensionsWidget.layout().addWidget(self.sceneUnitsLabel)

        self.pifDimensionsGroupBox.layout().addWidget(self.sceneDimensionsWidget)

        lTheCellSceneLayerTabLayout.addWidget(self.pifDimensionsGroupBox)


        # ----------------------------------------------------------------
        #
        # QWidget setup (2f) - add controls for scene scale/zoom, and for
        #       the types of regions and cells:
        #






#         self.mainControlPanelLayout.addStretch(10)



        # ----------------------------------------------------------------
        #
        # QWidget setup (3) - prepare for a QTabView for Image Layer controls:
        #
        #
        lTheImageLayerTab = QtGui.QWidget()
        lTheImageLayerTabLayout = QtGui.QVBoxLayout()
        lTheImageLayerTabLayout.setMargin(2)
        lTheImageLayerTabLayout.setSpacing(4)
        lTheImageLayerTabLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        lTheImageLayerTab.setLayout(lTheImageLayerTabLayout)





        # ----------------------------------------------------------------
        #
        # QWidget setup (3a) - add file labels and controls for Image Layer content:
        #
        imageHeaderLabel = QtGui.QLabel("Image Layer")
        imageHeaderLabel.setFont(labelHeaderFont)
        imageHeaderLabel.setMargin(2)
        imageHeaderLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lTheImageLayerTabLayout.addWidget(imageHeaderLabel)

        # this QGroupBox is defined in its own class:
        lTheImageLayerTabLayout.addWidget(self.controlsForInputImagePicking)


#         self.pifWidthXLabel = QtGui.QLabel("PIFF grid width: x=")
#         self.pifWidthXInputBox = QtGui.QSpinBox()
#         self.pifWidthXInputBox.setRange(0, 2000)
#         lTheImageLayerTabLayout.addWidget(self.pifWidthXLabel)
#         lTheImageLayerTabLayout.addWidget(self.pifWidthXInputBox)
#
#         self.pifHeightYLabel = QtGui.QLabel("PIFF grid height: y=")
#         self.pifHeightYInputBox = QtGui.QSpinBox()
#         self.pifHeightYInputBox.setRange(0, 2000)
#         lTheImageLayerTabLayout.addWidget(self.pifHeightYLabel)
#         lTheImageLayerTabLayout.addWidget(self.pifHeightYInputBox)

        # lTheImageLayerTabLayout.addSeparator()
        # lTheImageLayerTabLayout.addStretch(10)




        # ----------------------------------------------------------------
        #
        # QWidget setup (4) - prepare for a QTabView for Image Sequence controls:
        #
        #
        lTheImageSequenceTab = QtGui.QWidget()
        lTheImageSequenceTabLayout = QtGui.QVBoxLayout()
        lTheImageSequenceTabLayout.setMargin(2)
        lTheImageSequenceTabLayout.setSpacing(4)
        lTheImageSequenceTabLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        lTheImageSequenceTab.setLayout(lTheImageSequenceTabLayout)

        # ----------------------------------------------------------------
        #
        # QWidget setup (4a) - add controls for Image Sequence content:
        #
        lImageSequenceHeaderLabel = QtGui.QLabel("Image Sequence")
        lImageSequenceHeaderLabel.setFont(labelHeaderFont)
        lImageSequenceHeaderLabel.setMargin(2)
        lImageSequenceHeaderLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lTheImageSequenceTabLayout.addWidget(lImageSequenceHeaderLabel)

        # this QGroupBox is defined in its own class:
        lTheImageSequenceTabLayout.addWidget(self.controlsForImageSequence)
        # lTheImageSequenceTabLayout.addStretch(10)







        # ----------------------------------------------------------------
        #
        # QWidget setup (4) - prepare for a QTabView for Clusters controls:
        #
        #
        lTheClustersTab = QtGui.QWidget()
        lTheClustersTabLayout = QtGui.QVBoxLayout()
        lTheClustersTabLayout.setMargin(2)
        lTheClustersTabLayout.setSpacing(4)
        lTheClustersTabLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        lTheClustersTab.setLayout(lTheClustersTabLayout)

        # ----------------------------------------------------------------
        #
        # QWidget setup (4a) - add controls for Image Sequence content:
        #
        lClustersHeaderLabel = QtGui.QLabel("Cell Clusters")
        lClustersHeaderLabel.setFont(labelHeaderFont)
        lClustersHeaderLabel.setMargin(2)
        lClustersHeaderLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lTheClustersTabLayout.addWidget(lClustersHeaderLabel)

        # this QGroupBox is defined in its own class:
        lTheClustersTabLayout.addWidget(self.controlsForClusters)
        # lTheClustersTabLayout.addStretch(10)








        # ----------------------------------------------------------------
        #
        # QWidget setup - add a QGroupBox for showing Image Sequence dimensions:
        #
        self.imageSequenceDimensionsGroupBox = QtGui.QGroupBox("Image Sequence Dimensions")
        self.imageSequenceDimensionsGroupBox.setLayout(QtGui.QVBoxLayout())
        self.imageSequenceDimensionsGroupBox.layout().setMargin(2)
        self.imageSequenceDimensionsGroupBox.layout().setSpacing(4)
        self.imageSequenceDimensionsGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # 2011 - Mitja: add a widget displaying the imageSequence dimensions at all times:
        self.imageSequenceWidthLabel = QtGui.QLabel()
        self.imageSequenceWidthLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.imageSequenceWidthLabel.setText("w")
        self.imageSequenceWidthLabel.setFont(lFont)
        self.imageSequenceWidthLabel.setMargin(2)
        imageSequenceTimesSignLabel = QtGui.QLabel()
        imageSequenceTimesSignLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        imageSequenceTimesSignLabel.setText( u"\u00D7" ) # <-- the multiplication sign as unicode
        imageSequenceTimesSignLabel.setFont(lFont)
        imageSequenceTimesSignLabel.setMargin(2)
        self.imageSequenceHeightLabel = QtGui.QLabel()
        self.imageSequenceHeightLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.imageSequenceHeightLabel.setText("h")
        self.imageSequenceHeightLabel.setFont(lFont)
        self.imageSequenceHeightLabel.setMargin(2)
        self.imageSequenceUnitsLabel = QtGui.QLabel()
        self.imageSequenceUnitsLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.imageSequenceUnitsLabel.setText("  pixel")
        self.imageSequenceUnitsLabel.setFont(lFont)
        self.imageSequenceUnitsLabel.setMargin(2)

        self.imageSequenceDepthLabel = QtGui.QLabel()
        self.imageSequenceDepthLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.imageSequenceDepthLabel.setText("d")
        self.imageSequenceDepthLabel.setFont(lFont)
        self.imageSequenceDepthLabel.setMargin(2)
        self.imageSequenceImageOrImagesLabel = QtGui.QLabel()
        self.imageSequenceImageOrImagesLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.imageSequenceImageOrImagesLabel.setText( "  images" )
        self.imageSequenceImageOrImagesLabel.setFont(lFont)
        self.imageSequenceImageOrImagesLabel.setMargin(2)


        imageSequenceXYDimLayout = QtGui.QHBoxLayout()
        imageSequenceXYDimLayout.setMargin(2)
        imageSequenceXYDimLayout.setSpacing(4)
        imageSequenceXYDimLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        imageSequenceZDimLayout = QtGui.QHBoxLayout()
        imageSequenceZDimLayout.setMargin(2)
        imageSequenceZDimLayout.setSpacing(4)
        imageSequenceZDimLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        imageSequenceXYDimLayout.addWidget(self.imageSequenceWidthLabel)
        imageSequenceXYDimLayout.addWidget(imageSequenceTimesSignLabel)
        imageSequenceXYDimLayout.addWidget(self.imageSequenceHeightLabel)
        imageSequenceXYDimLayout.addWidget(self.imageSequenceUnitsLabel)

        imageSequenceZDimLayout.addWidget(self.imageSequenceDepthLabel)
        imageSequenceZDimLayout.addWidget(self.imageSequenceImageOrImagesLabel)


        self.imageSequenceDimensionsGroupBox.layout().addLayout(imageSequenceXYDimLayout)
        self.imageSequenceDimensionsGroupBox.layout().addLayout(imageSequenceZDimLayout)

        lTheImageSequenceTabLayout.addWidget(self.imageSequenceDimensionsGroupBox)









        # ----------------------------------------------------------------
        #
        # QWidget setup - add a QGroupBox for showing Image Sequence index and filename:
        #
        self.imageSequenceFilenamesGroupBox = QtGui.QGroupBox("Image Sequence Files")
        self.imageSequenceFilenamesGroupBox.setLayout(QtGui.QVBoxLayout())
        self.imageSequenceFilenamesGroupBox.layout().setMargin(2)
        self.imageSequenceFilenamesGroupBox.layout().setSpacing(4)
        self.imageSequenceFilenamesGroupBox.layout().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # 2011 - Mitja: add a widget displaying the imageSequence dimensions at all times:
        imageSequenceFileFirstLabel = QtGui.QLabel()
        imageSequenceFileFirstLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        imageSequenceFileFirstLabel.setText( "File index:" )
        imageSequenceFileFirstLabel.setFont(lFont)
        imageSequenceFileFirstLabel.setMargin(2)
        self.imageSequenceFileCurrentIndexLabel = QtGui.QLabel()
        self.imageSequenceFileCurrentIndexLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.imageSequenceFileCurrentIndexLabel.setText(" ")
        self.imageSequenceFileCurrentIndexLabel.setFont(lFont)
        self.imageSequenceFileCurrentIndexLabel.setMargin(2)
        imageSequenceFileSecondLabel = QtGui.QLabel()
        imageSequenceFileSecondLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        imageSequenceFileSecondLabel.setText( "File name:" )
        imageSequenceFileSecondLabel.setFont(lFont)
        imageSequenceFileSecondLabel.setMargin(2)
        self.imageSequenceCurrentFilename = QtGui.QLabel()
        self.imageSequenceCurrentFilename.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.imageSequenceCurrentFilename.setText(" ")
        self.imageSequenceCurrentFilename.setFont(lFont)
        self.imageSequenceCurrentFilename.setMargin(2)



        imageSequenceCurrentFileIndexLayout = QtGui.QHBoxLayout()
        imageSequenceCurrentFileIndexLayout.setMargin(2)
        imageSequenceCurrentFileIndexLayout.setSpacing(4)
        imageSequenceCurrentFileIndexLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        imageSequenceCurrentFileNameLayout = QtGui.QHBoxLayout()
        imageSequenceCurrentFileNameLayout.setMargin(2)
        imageSequenceCurrentFileNameLayout.setSpacing(4)
        imageSequenceCurrentFileNameLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)


        imageSequenceCurrentFileIndexLayout.addWidget(imageSequenceFileFirstLabel)
        imageSequenceCurrentFileIndexLayout.addWidget(self.imageSequenceFileCurrentIndexLabel)
        imageSequenceCurrentFileNameLayout.addWidget(imageSequenceFileSecondLabel)
        imageSequenceCurrentFileNameLayout.addWidget(self.imageSequenceCurrentFilename)



        self.imageSequenceFilenamesGroupBox.layout().addLayout(imageSequenceCurrentFileIndexLayout)
        self.imageSequenceFilenamesGroupBox.layout().addLayout(imageSequenceCurrentFileNameLayout)

        lTheImageSequenceTabLayout.addWidget(self.imageSequenceFilenamesGroupBox)











        # ----------------------------------------------------------------
        #
        # QWidget setup (final) - add each portion of controls to its own tab:
        #

        self.theMainTabWidget = QtGui.QTabWidget()
        # first hide the entire tab while it's populated, to avoid flickering:
        self.theMainTabWidget.hide()
#        self.theMainTabWidget.setTabShape(QtGui.QTabWidget.Triangular)
        self.theMainTabWidget.setTabsClosable(False)
        self.theMainTabWidget.setUsesScrollButtons(True)

        self.theMainTabWidget.insertTab(0, lTheCellSceneLayerTab, "Scene")
        self.theMainTabWidget.setTabToolTip(0, "Scene Layer - edit cells and regions in the Cell Scene")
        self.theMainTabWidget.setTabWhatsThis(0, "Scene Layer: create and edit cells and regions in the Cell Scene.")

        self.theMainTabWidget.insertTab(1, lTheImageLayerTab, "Image")
        self.theMainTabWidget.setTabToolTip(1, "Image Layer: pick color regions, draw scene regions on top of image")
        self.theMainTabWidget.setTabWhatsThis(1, "Image Layer: pick color regions, draw scene regions on top of image.")

        self.theMainTabWidget.insertTab(2, lTheImageSequenceTab, "Sequence")
        self.theMainTabWidget.setTabToolTip(2, "Image Sequence: access a stack of images in a sequence")
        self.theMainTabWidget.setTabWhatsThis(2, "Image Sequence: access a stack of images in a sequence.")

# 2012 - Mitja: todo SUPPORT drawing cells in a cluster, uncomment this tab and restart:
#
#         self.theMainTabWidget.insertTab(3, lTheClustersTab, "Clusters")
#         self.theMainTabWidget.setTabToolTip(2, "Edit Cluster: draw cells in a cluster")
#         self.theMainTabWidget.setTabWhatsThis(2, "Edit Cluster: draw cells in a cluster.")



        # call handleCurrentTabChanged() every time a new tab is selected in self.theMainTabWidget
        self.theMainTabWidget.currentChanged[int].connect( \
            self.handleCurrentTabChanged)


        self.theMainTabWidget.show()

        self.mainControlPanelLayout.addWidget( self.theMainTabWidget ) 

# NOW ADD A GUI for the grid and the controls here,
# and a GUI for the image controls on the toolbar on the window!


        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: finishSetup(): done" )+" ", CDConstants.DebugTODO )
    # end of def finishSetup()
    # ------------------------------------------------------------------




    # ------------------------------------------------------------
    # 2011 Mitja - slot method handling "currentChanged" events
    #    (AKA signals) arriving from self.theMainTabWidget:
    # ------------------------------------------------------------
    def handleCurrentTabChanged(self, pIndex):
        CDConstants.printOut("CDControlPanel - handleCurrentTabChanged(), pIndex = " +str(pIndex), CDConstants.DebugExcessive)
        if (pIndex == 0):
            self.controlsForLayerSelection.clickOnButton( \
                CDConstants.SceneModeMoveItem)
        elif (pIndex == 1):
            self.controlsForLayerSelection.clickOnButton( \
                CDConstants.SceneModeImageLayer)
        elif (pIndex == 2):
            self.controlsForLayerSelection.clickOnButton( \
                CDConstants.SceneModeImageSequence)
        elif (pIndex == 3):
            self.controlsForLayerSelection.clickOnButton( \
                CDConstants.SceneModeEditCluster)
        else:
            CDConstants.printOut("CDControlPanel - handleCurrentTabChanged(), pIndex = " +str(pIndex)+" clicked on nonexistant tab!", CDConstants.DebugImportant)



    # ------------------------------------------------------------
    # programmatically set the current tab in the QTabWidget:
    # ------------------------------------------------------------
    def setCurrentTab(self, pId):
        self.theMainTabWidget.setCurrentIndex(pId)






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
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: miCreateSlider(): done" )+" ", CDConstants.DebugTODO )


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
        layout.setMargin(2)
        layout.addWidget(button, 0, 0, QtCore.Qt.AlignHCenter)
        # layout.addWidget(QtGui.QLabel(pText), 1, 0, QtCore.Qt.AlignCenter)

        widget = QtGui.QWidget()
        widget.setLayout(layout)

        return widget
        # return button
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: createRegionShapeButton(): done" )+" ", CDConstants.DebugTODO )



    # ------------------------------------------------------------
    def createBackgroundCellWidget(self, text, pPixmap):
        button = QtGui.QToolButton()
        button.setText(text)
        button.setIcon(QtGui.QIcon(pPixmap))
        button.setIconSize(QtCore.QSize(24, 24))
        button.setCheckable(True)
        self.buttonGroupForBackgrounds.addButton(button)

        layout = QtGui.QGridLayout()
        layout.setMargin(2)
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
        layout.setMargin(2)
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
        if self.isTheBackgroundAnImage() == True:
            if type(self.parentWindow).__name__ == "CDDiagramSceneMainWidget":
                self.parentWindow.updateSceneBackgroundImage(self.imageNameFromFile)
                # CDConstants.printOut( " "+str( "self.parentWindow.updateSceneBackgroundImage(self.imageNameFromFile) DONE" )+" ", CDConstants.DebugTODO )

        # 2011 - Mitja: also update the button for selecting the Image Layer in the editor:
        self.controlsForLayerSelection.setImageLayerButtonIcon( QtGui.QIcon( QtGui.QPixmap.fromImage(pImage) ) )


        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: updateBackgroundImageButtons() done." )+" ", CDConstants.DebugTODO )



    # ------------------------------------------------------------
    # 2010 - Mitja - add a fuction to quickly find out whether the scene background
    #     is set to a user-loaded image or to one of the plain built-in patterns:
    # ------------------------------------------------------------
    def isTheBackgroundAnImage(self):
        buttons = self.buttonGroupForBackgrounds.buttons()
        for myButton in buttons:
            CDConstants.printOut( " 123123123123123 = "+str(myButton)+" button has text "+str(myButton.text())+" ", CDConstants.DebugTODO )
            if myButton.isChecked() == True:
                lText = myButton.text()

        if (lText == "Blue Grid") or (lText == "White Grid") or (lText == "Gray Grid") or (lText == "No Grid"):
            return False
        else:
            return True

        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: isTheBackgroundAnImage() done." )+" ", CDConstants.DebugTODO )



    # ------------------------------------------------------------
    def setWidgetIcon(self, pDiagramType, pIcon):
        self.widgetDict[pDiagramType] = pIcon
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setWidgetIcon(): done" )+" ", CDConstants.DebugTODO )



    # ------------------------------------------------------------
    def setControlsForDrawingRegionOrCellToggle(self, pWidget):
        self.controlsForDrawingRegionOrCell = pWidget
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setControlsForDrawingRegionOrCellToggle(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------
    def setControlsForTypes(self, pWidget):
        self.controlsForTypes = pWidget
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setControlsForTypes(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------
    def setControlsForLayerSelection(self, pWidget):
        self.controlsForLayerSelection = pWidget
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setControlsForLayerSelection(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------
    def setControlsForSceneScaleZoom(self, pWidget):
        self.controlsForSceneScaleZoom = pWidget
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setControlsForSceneScaleZoom(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------
    def setControlsForInputImagePicking(self, pGroupBox):
        self.controlsForInputImagePicking = pGroupBox
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setControlsForInputImagePicking(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------
    def setControlsForImageSequence(self, pGroupBox):
        self.controlsForImageSequence = pGroupBox
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setControlsForImageSequence(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------
    def setControlsForClusters(self, pGroupBox):
        self.controlsForClusters = pGroupBox
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setControlsForClusters(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------
    def setControlsForSceneItemEdit(self, pGroupBox):
        self.controlsForSceneItemEdit = pGroupBox
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setControlsForSceneItemEdit(): done" )+" ", CDConstants.DebugTODO )








    # ------------------------------------------------------------
    def setButtonGroupForRegionShapes(self, pButtonGroup):
        self.buttonGroupForRegionShapes = pButtonGroup
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setButtonGroupForRegionShapes(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------
    def setButtonGroupForBackgrounds(self, pButtonGroup):
        self.buttonGroupForBackgrounds = pButtonGroup
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setButtonGroupForRegionShapes(): done" )+" ", CDConstants.DebugTODO )




    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def setImageFileNameLabel(self, pImageFileName):
        # save the image file name into the label to show it in the panel
        self.imageFileNameLabel.setText(pImageFileName)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setImageFileNameLabel(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def setPiffFileNameLabel(self, pPifFileName):
        # save the PIFF file name into the label to show it in the panel
        self.piffFileNameLabel.setText(pPifFileName)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setPiffFileNameLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
#     def setWindowFlags(self, pFlags):
#         super(CDControlPanel, self).setWindowFlags(pFlags)
#         # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setWindowFlags(): done" )+" ", CDConstants.DebugTODO )






    # ------------------------------------------------------------------
    # functions for externally setting the control panel's label values:

    # ------------------------------------------------------------------
    def setSceneWidthLabel(self, pSceneWidthLabel):
        self.sceneWidthLabel.setText(pSceneWidthLabel)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setSceneWidthLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setSceneHeightLabel(self, pSceneHeightLabel):
        self.sceneHeightLabel.setText(pSceneHeightLabel)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setSceneHeightLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setSceneDepthLabel(self, pSceneDepthLabel):
        self.sceneDepthLabel.setText(pSceneDepthLabel)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setSceneHeightLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setSceneUnitsLabel(self, pSceneUnitsLabel):
        self.sceneUnitsLabel.setText(pSceneUnitsLabel)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setSceneUnitsLabel(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------------
    def setResizingItemXLabel(self, pLabelText):
        self.resizingItemXLabel.setText(pLabelText)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setResizingItemXLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setResizingItemYLabel(self, pLabelText):
        self.resizingItemYLabel.setText(pLabelText)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setResizingItemYLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setResizingItemWidthLabel(self, pLabelText):
        self.resizingItemWidthLabel.setText(pLabelText)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setResizingItemWidthLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setResizingItemHeightLabel(self, pLabelText):
        self.resizingItemHeightLabel.setText(pLabelText)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlPanel: setResizingItemHeightLabel(): done" )+" ", CDConstants.DebugTODO )




    # ------------------------------------------------------------------
    def setImageSequenceWidthLabel(self, pImageSequenceWidthLabel):
        self.imageSequenceWidthLabel.setText(pImageSequenceWidthLabel)
        CDConstants.printOut("___ - DEBUG ----- CDControlPanel: setImageSequenceWidthLabel(pImageSequenceWidthLabel=="+str(pImageSequenceWidthLabel)+"): done", CDConstants.DebugVerbose )

    # ------------------------------------------------------------------
    def setImageSequenceHeightLabel(self, pImageSequenceHeightLabel):
        self.imageSequenceHeightLabel.setText(pImageSequenceHeightLabel)
        CDConstants.printOut("___ - DEBUG ----- CDControlPanel: setImageSequenceHeightLabel(pImageSequenceHeightLabel=="+str(pImageSequenceHeightLabel)+"): done", CDConstants.DebugVerbose )

    # ------------------------------------------------------------------
    # setImageSequenceDepthLabel() accepts as input only a string containing a properly formed integer!
    # ------------------------------------------------------------------
    def setImageSequenceDepthLabel(self, pImageSequenceDepthLabel):
        self.imageSequenceDepthLabel.setText(pImageSequenceDepthLabel)
        # also update the image selection range in the Image Sequence controls,
        #   remembering that the index starts at 0 so the max image index should be:
        self.controlsForImageSequence.setMaxImageIndex(int(pImageSequenceDepthLabel)-1)
        CDConstants.printOut("___ - DEBUG ----- CDControlPanel: setImageSequenceDepthLabel(pImageSequenceDepthLabel=="+str(pImageSequenceDepthLabel)+"): done", CDConstants.DebugVerbose )

    # ------------------------------------------------------------------
    def setImageSequenceUnitsLabel(self, pImageSequenceUnitsLabel):
        self.imageSequenceUnitsLabel.setText(pImageSequenceUnitsLabel)
        CDConstants.printOut("___ - DEBUG ----- CDControlPanel: setImageSequenceUnitsLabel(pImageSequenceUnitsLabel=="+str(pImageSequenceUnitsLabel)+"): done", CDConstants.DebugVerbose )

    # ------------------------------------------------------------------
    def setImageSequenceImageUnitsLabel(self, pImageSequenceImageUnitsLabel):
        self.imageSequenceImageOrImagesLabel.setText(pImageSequenceImageUnitsLabel)
        CDConstants.printOut("___ - DEBUG ----- CDControlPanel: setImageSequenceImageUnitsLabel(pImageSequenceImageUnitsLabel=="+str(pImageSequenceImageUnitsLabel)+"): done", CDConstants.DebugVerbose )



    # ------------------------------------------------------------------
    def setImageSequenceCurrentIndex(self, pImageSequenceCurrentIndex):
        self.imageSequenceFileCurrentIndexLabel.setText(pImageSequenceCurrentIndex)
        CDConstants.printOut("___ - DEBUG ----- CDControlPanel: setImageSequenceCurrentIndex(pImageSequenceCurrentIndex=="+str(pImageSequenceCurrentIndex)+"): done", CDConstants.DebugVerbose )

    # ------------------------------------------------------------------
    def setImageSequenceCurrentFilename(self, pImageSequenceCurrentFilename):
        self.imageSequenceCurrentFilename.setText(pImageSequenceCurrentFilename)
        CDConstants.printOut("___ - DEBUG ----- CDControlPanel: setImageSequenceCurrentFilename(pImageSequenceCurrentFilename=="+str(pImageSequenceCurrentFilename)+"): done", CDConstants.DebugVerbose )



# end class CDControlPanel(QtGui.QWidget)
# ======================================================================




# Local Variables:
# coding: US-ASCII
# End:
