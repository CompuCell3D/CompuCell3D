#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ======================================================================
# a QWidget-based control panel, in application-specific panel style
# ======================================================================

class CDControlCellScene(QtGui.QWidget):

    # possible types for regions used in the diagram scene:
    RectangleConst, TenByTenBoxConst, StartEndConst, \
        TwoByTwoBoxConst, PathConst = range(5)
   
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def __init__(self, pParent = None):
        super(CDControlCellScene,self).__init__(pParent)

        # ----- ----- ----- -----
        # declare object globals:
        # ----- ----- ----- -----
        self.parentWindow = pParent

        # a QGroupBox containing buttons for toggling the layer selection
        #    it's assigned below, in setControlsForLayerSelection()
        self.controlsForLayerSelection = 0

# 
#         # a QGroupBox with a combobox (pop-up menu) for scene zoom
#         #    it's assigned below, in setControlsForSceneZoom()
#         self.controlsForSceneZoom = 0

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
            CDControlCellScene.RectangleConst: QtGui.QIcon(), \
            CDControlCellScene.TenByTenBoxConst: QtGui.QIcon(), \
            CDControlCellScene.StartEndConst: QtGui.QIcon(), \
            CDControlCellScene.TwoByTwoBoxConst: QtGui.QIcon(), \
            CDControlCellScene.PathConst: QtGui.QIcon(), \
            }

        # a QButtonGroup containing all buttons for region shapes,
        #    it's assigned below, in setButtonGroupForRegionShapes()
        self.buttonGroupForRegionShapes = 0

        # a QButtonGroup containing all buttons for backgrounds,
        #    it's assigned below, in setButtonGroupForBackgrounds()
        self.buttonGroupForBackgrounds = 0




        # a QGroupBox containing buttons and sliders for controlling
        #    clusters of cells, it's assigned below, in setControlsForClusters()
        self.controlsForClusters = 0


        # class globals for the background button from input image:
        # 2010 - Mitja: add code for new backgrounds:
        lBoringPixMap = QtGui.QPixmap(240, 180)
        lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.white) )
        self.imageFromFile = QtGui.QImage(lBoringPixMap)
        self.imageNameFromFile = "BlankBackground"

        # ----- ----- ----- -----
        # declare object globals:
        # ----- ----- ----- -----
        #
        # QWidget setup (1) - windowing GUI setup for Control Panel:
        #

        self.setWindowTitle("Control Panel Window Title")
        # QVBoxLayout layout lines up widgets vertically:
        self.mainControlCellSceneLayout = QtGui.QVBoxLayout()
        self.mainControlCellSceneLayout.setMargin(4)
        self.mainControlCellSceneLayout.setSpacing(4)
        self.mainControlCellSceneLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.setLayout(self.mainControlCellSceneLayout)


        #
        # QWidget setup (2) - more windowing GUI setup for Control Panel:
        #

        CDConstants.printOut( "___ - DEBUG ----- CDControlCellScene: __init__(): done" , CDConstants.DebugTODO )

# 
#     # ------------------------------------------------------------------
#     def sizeHint(self):
#         return  QtCore.QSize(268, 528)



    # ------------------------------------------------------------------
    def populateControlPanel(self):


        # ----------------------------------------------------------------
        #
        # QWidget setup (2) - prepare Scene Layer controls:
        #


# 
#         # ----------------------------------------------------------------
#         #
#         # QWidget setup (2a) - add controls for scene zoom:
#         #
#         lFirstSimpleQHBoxLayout = QtGui.QHBoxLayout()
#         lFirstSimpleQHBoxLayout.setMargin(2)
#         lFirstSimpleQHBoxLayout.setSpacing(4)
#         lFirstSimpleQHBoxLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
# 
#         # 2012 - Mitja: the layer selection controls have been moved to a QToolBar:
#         #    but we keep them here until we decouple actions/signals/slots/handlers:
#         #
#         #    the layer selection control is defined in its own class:
#         #
#         #    lFirstSimpleQHBoxLayout.addWidget(self.controlsForLayerSelection)
# 
#         # the combobox/pop-up menu for scene zoom control is defined in its own class:
#         #
#         lFirstSimpleQHBoxLayout.addWidget(self.controlsForSceneZoom)
# 
#         self.mainControlCellSceneLayout.addLayout(lFirstSimpleQHBoxLayout)
# 
# 





        # ----------------------------------------------------------------
        #
        # QWidget setup (2a) - add controls for Item editing in the Scene:
        #

        # ----------------------------------------------------------------
        #
        # QWidget setup (2a) - add a QGroupBox containing buttons with
        #    scene item edit controls, such as cut/copy/paste/delete etc.
        #
        # this is the "Item Edit" QGroupBox, defined in TODO TODO WHERE? .py file? :
        #
        self.mainControlCellSceneLayout.addWidget(self.controlsForSceneItemEdit)




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

        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: 1" )+" ", CDConstants.DebugTODO )
        sceneItemLayerLayout.addWidget(self.createRegionShapeButton("Ellipse", CDControlCellScene.PathConst))

        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: 2" )+" ", CDConstants.DebugTODO )
        sceneItemLayerLayout.addWidget(self.createRegionShapeButton("Rectangle", CDControlCellScene.RectangleConst))

        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: empty" )+" ", CDConstants.DebugTODO )
        sceneItemLayerLayout.addSpacing(40)

        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: 3" )+" ", CDConstants.DebugTODO )
        sceneItemLayerLayout.addWidget(self.createRegionShapeButton("TenByTenBox", CDControlCellScene.TenByTenBoxConst))

        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: 4" )+" ", CDConstants.DebugTODO )
        sceneItemLayerLayout.addWidget(self.createRegionShapeButton("TwoByTwoBox", CDControlCellScene.TwoByTwoBoxConst))

       
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

        self.mainControlCellSceneLayout.addWidget(self.sceneItemLayerGroupBox)



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

        # add both these two controls as hbox to the main controls layout:
        self.mainControlCellSceneLayout.addLayout(anotherSimpleQHBoxLayout)


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
#         self.backgroundLayout.addWidget(self.createBackgroundCellWidget("White Grid",
#                 ':/icons/background2.png'), 0, 2)
#         self.backgroundLayout.addWidget(self.createBackgroundCellWidget("Gray Grid",
#                 ':/icons/background3.png'), 0, 3)
#         self.backgroundLayout.addWidget(self.createBackgroundCellWidget("Blue Grid",
#                 ':/icons/background1.png'), 0, 4)

        # 2011 - Mitja: make sure that one background button is checked:
        theBgButtons = self.buttonGroupForBackgrounds.buttons()
        for myButton in theBgButtons:
            CDConstants.printOut( "123123123123123    myButton="+str(myButton)+"myButton.text()="+str(myButton.text()), CDConstants.DebugTODO )
            if myButton.text() == "No Grid":
                myButton.setChecked(True)


        self.sceneBackgroundLayerGroupBox.layout().addLayout(self.backgroundLayout)

        self.mainControlCellSceneLayout.addWidget(self.sceneBackgroundLayerGroupBox)


        # ----------------------------------------------------------------
        #
        # QWidget setup (2e) - add a QGroupBox for showing Scene dimensions:
        #
        # this now goes to its own class: CDViewSceneDimensions()

        # ----------------------------------------------------------------
        #
        # QWidget setup (2f) - add controls for scene zoom, and for
        #       the types of regions and cells:
        #




        # ----------------------------------------------------------------
        #
        # QWidget setup (6a) - add controls for Cell Clusters content:
        #
        labelHeaderFont = QtGui.QFont()
        labelHeaderFont.setStyleStrategy(QtGui.QFont.PreferAntialias | QtGui.QFont.PreferQuality)
        labelHeaderFont.setStyleHint(QtGui.QFont.SansSerif)
        labelHeaderFont.setWeight(QtGui.QFont.Bold)
        lClustersHeaderLabel = QtGui.QLabel("Cell Clusters")
        lClustersHeaderLabel.setFont(labelHeaderFont)
        lClustersHeaderLabel.setMargin(2)
        lClustersHeaderLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # this QGroupBox is defined in its own class:
        # lTheClustersTabLayout.addWidget(self.controlsForClusters)








        CDConstants.printOut("___ - DEBUG ----- CDControlCellScene: populateControlPanel(): done", CDConstants.DebugTODO )
    # end of def populateControlPanel()
    # ------------------------------------------------------------------







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
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: miCreateSlider(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------
    def createRegionShapeButton(self, pText, pDiagramType):

        # 2010 - Mitja: add code for handling insertion of path-derived items:
        if (pDiagramType == CDControlCellScene.PathConst) :
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
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: createRegionShapeButton(): done" )+" ", CDConstants.DebugTODO )



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


        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: updateBackgroundImageButtons() done." )+" ", CDConstants.DebugTODO )



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

        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: isTheBackgroundAnImage() done." )+" ", CDConstants.DebugTODO )



    # ------------------------------------------------------------
    def setWidgetIcon(self, pDiagramType, pIcon):
        self.widgetDict[pDiagramType] = pIcon
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setWidgetIcon(): done" )+" ", CDConstants.DebugTODO )



    # ------------------------------------------------------------
    def setControlsForDrawingRegionOrCellToggle(self, pWidget):
        self.controlsForDrawingRegionOrCell = pWidget
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setControlsForDrawingRegionOrCellToggle(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------
    def setControlsForTypes(self, pWidget):
        self.controlsForTypes = pWidget
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setControlsForTypes(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------
    def setControlsForLayerSelection(self, pWidget):
        self.controlsForLayerSelection = pWidget
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setControlsForLayerSelection(): done" )+" ", CDConstants.DebugTODO )


# 
#     # ------------------------------------------------------------
#     def setControlsForSceneZoom(self, pWidget):
#         self.controlsForSceneZoom = pWidget
#         # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setControlsForSceneZoom(): done" )+" ", CDConstants.DebugTODO )



    # ------------------------------------------------------------
    def setControlsForClusters(self, pGroupBox):
        self.controlsForClusters = pGroupBox
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setControlsForClusters(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------
    def setControlsForSceneItemEdit(self, pGroupBox):
        self.controlsForSceneItemEdit = pGroupBox
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setControlsForSceneItemEdit(): done" )+" ", CDConstants.DebugTODO )








    # ------------------------------------------------------------
    def setButtonGroupForRegionShapes(self, pButtonGroup):
        self.buttonGroupForRegionShapes = pButtonGroup
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setButtonGroupForRegionShapes(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------
    def setButtonGroupForBackgrounds(self, pButtonGroup):
        self.buttonGroupForBackgrounds = pButtonGroup
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setButtonGroupForRegionShapes(): done" )+" ", CDConstants.DebugTODO )




    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def setImageFileNameLabel(self, pImageFileName):
        # save the image file name into the label to show it in the panel
        self.imageFileNameLabel.setText(pImageFileName)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setImageFileNameLabel(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def setPiffFileNameLabel(self, pPifFileName):
        # save the PIFF file name into the label to show it in the panel
        self.piffFileNameLabel.setText(pPifFileName)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setPiffFileNameLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
#     def setWindowFlags(self, pFlags):
#         super(CDControlCellScene, self).setWindowFlags(pFlags)
#         # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setWindowFlags(): done" )+" ", CDConstants.DebugTODO )






    # ------------------------------------------------------------------
    def setResizingItemXLabel(self, pLabelText):
        self.resizingItemXLabel.setText(pLabelText)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setResizingItemXLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setResizingItemYLabel(self, pLabelText):
        self.resizingItemYLabel.setText(pLabelText)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setResizingItemYLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setResizingItemWidthLabel(self, pLabelText):
        self.resizingItemWidthLabel.setText(pLabelText)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setResizingItemWidthLabel(): done" )+" ", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setResizingItemHeightLabel(self, pLabelText):
        self.resizingItemHeightLabel.setText(pLabelText)
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDControlCellScene: setResizingItemHeightLabel(): done" )+" ", CDConstants.DebugTODO )




# end class CDControlCellScene(QtGui.QWidget)
# ======================================================================




# Local Variables:
# coding: US-ASCII
# End:
