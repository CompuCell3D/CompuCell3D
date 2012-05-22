#!/usr/bin/env python
#
# CDSceneRasterizer - add-on QGraphicsScene rasterizer for CellDraw - Mitja 2010
#
# ------------------------------------------------------------

import sys     # for handling command-line arguments, remove in final version
import os      # for path and split functions
import shutil  # for copying files as if it were from the shell
import inspect # for debugging functions, remove in final version
#
import random  # for generating regions with random-filled cell types
#
import math    # for ceiling functions
#
import time    # for sleep()


import numpy   # to have arrays!


from PyQt4 import QtGui, QtCore
#

# -->  -->  --> mswat code removed to run in MS Windows --> -->  -->
# -->  -->  --> mswat code removed to run in MS Windows --> -->  -->
# from PyQt4 import Qt
# <--  <--  <-- mswat code removed to run in MS Windows <-- <--  <--
# <--  <--  <-- mswat code removed to run in MS Windows <-- <--  <--

# -->  -->  --> mswat code added to run in MS Windows --> -->  -->
# -->  -->  --> mswat code added to run in MS Windows --> -->  -->
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4
# <--  <--  <-- mswat code added to run in MS Windows <-- <--  <--
# <--  <--  <-- mswat code added to run in MS Windows <-- <--  <--



# debugging functions, remove in final Panel version
def debugWhoIsTheRunningFunction():
    return inspect.stack()[1][3]
def debugWhoIsTheParentFunction():
    return inspect.stack()[2][3]
def gigi():
    print "hello, I'm %s, parent is %s" % \
          (debugWhoIsTheRunningFunction(), debugWhoIsTheParentFunction())
    pigi()
def pigi():
    print "hello, I'm %s, parent is %s" % \
          (debugWhoIsTheRunningFunction(), debugWhoIsTheParentFunction())


# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants




# ------------------------------------------------------------
# a class to store a pixmap image, based on PIF_Generator's
#     original code for InputImageLabel()
# ------------------------------------------------------------
class RasterizedImageLabel(QtGui.QLabel):
    # 2010 - Mitja: for unexplained reasons, this class is based on QLabel,
    #   even though it is NOT used as a label.
    #   Instead, this class draws an image, intercepts
    #   mouse click events, etc.

    def __init__(self,parent=None):
        QtGui.QLabel.__init__(self, parent)
        QtGui.QWidget.__init__(self, parent)
        self.scaleGrid = 1.0
        self.width = 240
        self.height = 180
        self.rasterWidth = 10
        # the "image" QImage is the one we see, i.e. the rasterized one
        self.image = QtGui.QImage()
        self.x = 0
        self.y = 0
        self.fixedSizeRaster = False

    def paintEvent(self, event):

        # QtGui.QLabel.paintEvent(self,event)

        # start a QPainter on this QLabel - this is why we pass "self" as paramter:
        lPainter = QtGui.QPainter(self)

        # take care of the RHS <-> LHS mismatch at its visible end,
        #   by flipping the y coordinate in the QPainter's affine transformations:       
        lPainter.translate(0.0, float(self.pixmap().height()))
        lPainter.scale(1.0, -1.0)

        # access the QLabel's pixmap to draw it explicitly, using QPainter's scaling:
        lPainter.drawPixmap(0, 0, self.pixmap())

        if self.fixedSizeRaster is True:
            lPen = QtGui.QPen()
            lPen.setColor(QtGui.QColor(QtCore.Qt.black))
            lPen.setWidth(1)
            lPen.setCosmetic(True)
            lPainter.setPen(lPen)
            self.drawGrid(lPainter)
        else:
            # we don't need to draw the grid on top of the label:
            pass


    def drawGrid(self,painter):
        for x in xrange(0, self.width, self.rasterWidth):
            #draw.line([(x, 0), (x, h)], width=2, fill='#000000')
            painter.drawLine(x,0,x,self.height)
        for y in xrange(0, self.height, self.rasterWidth):
         #draw.line([(0, y), (w, y)], width=2, fill='#000000')
            painter.drawLine(0,y,self.width,y)


    def plotRect(self, pRGBA, pXmin, pYmin, pXmax, pYmax):

        lColor = QtGui.QColor()
        lColor.setRgba(pRGBA)

        lPen = QtGui.QPen()
        lPen.setColor(lColor)
        lPen.setWidth(1)
        lPen.setCosmetic(True)

        lPainter = QtGui.QPainter()
        lPainter.begin(self.pixmap())
        lPainter.setPen(lPen)

        if (pXmin >= pXmax) or (pYmin >= pYmax) :
            # if passed an incorrect rectangle (with max point < min point)
            # then just draw a 3x3 square around the min point
            lPainter.drawRect(pXmin-1, pYmin-1, 3, 3)

        else:
            lPainter.drawRect(pXmin, pYmin, (pXmax-pXmin), (pYmax-pYmin))
   
#             lPen.setColor(QtGui.QColor(QtCore.Qt.black))
#             lPen.setWidth(1)
#             lPen.setCosmetic(True)
#    
#             lPainter.setPen(lPen)
#             lPainter.drawRect(pXmin-1, pYmin-1, 3, 3)

        lPainter.end()


    def drawPixmapAtPoint(self, pPixmap, pXmin=0, pYmin=0):

        lPainter = QtGui.QPainter()
        lPainter.begin(self.pixmap())
        lPainter.drawPixmap(pXmin, pYmin, pPixmap)
        lPainter.end()
        self.update()

    def drawFixedSizeRaster(self, pFixedOrNot=False):
        self.fixedSizeRaster = pFixedOrNot
        self.update()


#     def mousePressEvent(self, event):
#         if event.button() == QtCore.Qt.LeftButton:
#             self.x = event.x()
#             self.y = event.y()
#             print "___ - DEBUG ----- RasterizedImageLabel: mousePressEvent() finds Color(x,y) = %s(%s,%s)" %(QtCore.QString("%1").arg(color, 8, 16), self.x, self.y)
#             self.emit(QtCore.SIGNAL("getpos()"))



# ======================================================================
# a QWidget-based widget panel, in application-specific panel style
# ======================================================================
class CDSceneRasterizer(QtGui.QWidget):

    def __init__(self, pParent=None):
        # it is compulsory to call the parent's __init__ class right away:
        super(CDSceneRasterizer, self).__init__(pParent)

        #
        # init (1) - windowing GUI stuff:
        #
        self.miInitGUI()

        #
        # init (2) - create widget with image-label, set it up and show it inside the panel:
        #
        self.layout().addWidget( self.miInitCentralImageLabelWidget() )

        #
        # init (3) - create empty region dict, color to name of region dict,
        #   and color to key of region dict:
        #
        self.regionsDict = dict()
        self.colorToNameRegionDict = dict()
        self.colorToCellSizeRegionDict = dict()
        self.colorToKeyRegionDict = dict()

        # 2011 - Mitja: globals storing fixed-size cell data for PIFF generation
        #
        # a numpy-based array global of size 1x1, it'll be resized when rasterizing:
        self.fixedSizeCellsArray = numpy.zeros( (1, 1), dtype=numpy.int )

        # two globals for width and height in fixed-sized cells,
        #   these will be computed from the scene size & cell size when rasterizing:
        self.fixedSizeWidthInCells = 0
        self.fixedSizeHeightInCells = 0

        # globals for ignoring white / black regions when saving the PIFF file:
        self.ignoreWhiteRegionsForPIF = False
        self.ignoreBlackRegionsForPIF = False
       
        # 2010 - Mitja: add functionality for saving PIFF metadata:
        self.savePIFMetadata = False

        # 2010 - Mitja: add a CellDraw preferences object,
        #   we'll get its object value in the setPreferencesObject() function defined below:
        self.cdPreferences = None


        # ---------------------------------------
        # 2011 - Mitja: add calling CC3D as subprocess:
        # ---------------------------------------
        # the globals required for calling CC3D will be set up in setupPathsToCC3D()
        # and the call to CC3D itself will be done by startCC3D()
        # ---------------------------------------
        self.cc3dPath = None
        self.cc3dPathAndStartupFileName = None
        self.cc3dOutputLocationPath = None
        
        # TODO TODO: is self.pluginObj necessary for anything?
        self.pluginObj = None

        self.cc3dProcess = None
        # ---------------------------------------

        CDConstants.printOut( "005 - DEBUG ----- CDSceneRasterizer: __init__(): done", CDConstants.DebugExcessive )


    # ------------------------------------------------------------------
    # define functions to initialize this panel:
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # init (1) - windowing GUI stuff:
    # ------------------------------------------------------------------
    def miInitGUI(self):

        # how will the CDSceneRasterizer look like:
        self.setWindowTitle("PIFF Output from Scene")
        # self.setMinimumSize(240, 180)
        # setGeometry is inherited from QWidget, taking 4 arguments:
        #   x,y  of the top-left corner of the QWidget, from top-left of screen
        #   w,h  of the QWidget
        # NOTE: the x,y is NOT the top-left edge of the window,
        #    but of its **content** (excluding the menu bar, toolbar, etc.
        # self.setGeometry(750,480,480,320)

        # QVBoxLayout layout lines up widgets vertically:
        self.setLayout(QtGui.QVBoxLayout())
        # self.layout().setAlignment(QtCore.Qt.AlignTop)
        self.layout().setAlignment = (QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.layout().setMargin(2)
        self.layout().setSpacing(2)

        #
        # QWidget setup (2) - more windowing GUI setup:
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


    # ------------------------------------------------------------------
    # init (3) - central widget containing an image within a label, set up and show:
    # ------------------------------------------------------------------
    def miInitCentralImageLabelWidget(self):

        # 2010 - Mitja: now create a QLabel, but NOT to be used as a label.
        #   as in the original PIF_Generator code:
        self.theRasterizedImageLabel = RasterizedImageLabel()

        # 2010 - Mitja: set the size policy of the theRasterizedImageLabel widget:.
        #   "The widget will get as much space as possible."
        self.theRasterizedImageLabel.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)

        # 2010 - Mitja: according to Qt documentation,
        #   "scale the pixmap to fill the available space" :
        self.theRasterizedImageLabel.setScaledContents(False)

        # self.theRasterizedImageLabel.setLineWidth(1)
        # self.theRasterizedImageLabel.setMidLineWidth(1)

        # set a QFrame type for this label, so that it shows up with a visible border around itself:
        self.theRasterizedImageLabel.setFrameShape(QtGui.QFrame.Panel)
        # self.theRasterizedImageLabel.setFrameShadow(QtGui.QFrame.Plain)

        # self.theRasterizedImageLabel.setAlignment = (QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        # self.theRasterizedImageLabel.setObjectName("theRasterizedImageLabel")

        # self.theRasterizedImageLabel.update()

        # -------------------------------------------
        # place theRasterizedImageLabel in the Panel's vbox layout:
        # one cell information widget, containing a vbox layout, in which to place it:
        theContainerWidget = QtGui.QWidget()

        # this infoLabel part is cosmetic and could safely be removed,
        #     unless useful info is provided here:
        self.infoLabel = QtGui.QLabel()
        self.infoLabel.setText("Generating PIFF elements from regions and cell scene...")
        self.infoLabel.setAlignment = (QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
#         self.infoLabel.setLineWidth(3)
#         self.infoLabel.setMidLineWidth(3)
#         self.infoLabel.setFrameStyle(QtGui.QFrame.StyledPanel | QtGui.QFrame.Sunken)

        # for unexplained reasons, a QLabel containing an image has to be placed
        #   in a QScrollArea to be displayed within a layout. Placing theRasterizedImageLabel
        #   directly in the layout would *not* display it at all!
        #   Therefore, create a QScrollArea and assign theRasterizedImageLabel to it:
        self.scrollArea = QtGui.QScrollArea()
        # self.scrollArea.setBackgroundRole(QtGui.QPalette.AlternateBase)
        self.scrollArea.setBackgroundRole(QtGui.QPalette.Mid)
        self.scrollArea.setWidget(self.theRasterizedImageLabel)
        self.scrollArea.setAlignment = (QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


        # create a progress bar:
        self.createProgressBar()      

        # create a layout and place all 'sub-widgets' in it:
        vbox = QtGui.QVBoxLayout()
        vbox.setMargin(2)
        vbox.setSpacing(4)
        vbox.setAlignment = (QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        vbox.addWidget(self.infoLabel)
        vbox.addWidget(self.progressBar)
        vbox.addWidget(self.scrollArea)

        # finally place the complete layout in a QWidget and return it:
        theContainerWidget.setLayout(vbox)
        return theContainerWidget


    # ------------------------------------------------------------------
    def setIgnoreWhiteRegions(self, pCheckBox):
        self.ignoreWhiteRegionsForPIF = pCheckBox
        CDConstants.printOut( ">>>>>>>>>>>>>>>>>>>>>>>> CDSceneRasterizer.ignoreWhiteRegionsForPIF is now =" + str(self.ignoreWhiteRegionsForPIF) , CDConstants.DebugVerbose )

    # ------------------------------------------------------------------
    def setIgnoreBlackRegions(self, pCheckBox):
        self.ignoreBlackRegionsForPIF = pCheckBox
        CDConstants.printOut( ">>>>>>>>>>>>>>>>>>>>>>>> CDSceneRasterizer.ignoreBlackRegionsForPIF is now =" + str(self.ignoreBlackRegionsForPIF), CDConstants.DebugVerbose )

    # ------------------------------------------------------------------
    def setRasterWidth(self, pWidth):
        self.theRasterizedImageLabel.rasterWidth = pWidth
        CDConstants.printOut( ">>>>>>>>>>>>>>>>>>>>>>>> CDSceneRasterizer.theRasterizedImageLabel.rasterWidth is now =" + \
          str(self.theRasterizedImageLabel.rasterWidth), CDConstants.DebugVerbose )

    # ------------------------------------------------------------------
    # 2010 - Mitja: add functionality for saving PIFF metadata:
    # ------------------------------------------------------------------
    def setSavePIFMetadata(self, pCheckBox):
        self.savePIFMetadata = pCheckBox
        CDConstants.printOut( ">>>>>>>>>>>>>>>>>>>>>>>> CDSceneRasterizer.savePIFMetadata is now =" + str(self.savePIFMetadata) , CDConstants.DebugVerbose )


    # ------------------------------------------------------------------
    # 2010 - Mitja: assign CellDraw preferences object:
    # ------------------------------------------------------------------
    def setPreferencesObject(self, pCDPreferences=None):
        self.cdPreferences = pCDPreferences
        CDConstants.printOut( ">>>>>>>>>>>>>>>>>>>>>>>> CDSceneRasterizer.cdPreferences is now =" + str(self.cdPreferences), CDConstants.DebugVerbose )




    # ------------------------------------------------------------------
    # assign a new graphics scene object to CDSceneRasterizer:
    # ------------------------------------------------------------------
    def setInputGraphicsScene(self, pGraphicsScene):

        # this QGraphicsScene is generated elsewhere (here we only need it
        #    to analyze its QGraphicsItems & compare them to the region/celltype table):
        #
        # TODO TODO TODO: declare the following globals in the __init__ method, for clearness!
        #
        self.theGraphicsScene = pGraphicsScene
        # the scene rect has been set (manually by us) to be the same as the PIFF output rect from user preferences:
        lSceneRect = self.theGraphicsScene.sceneRect()

        self.theSceneItemsRect = self.theGraphicsScene.itemsBoundingRect()
        self.theSceneLimitsMinX = self.theSceneItemsRect.left()
        self.theSceneLimitsMinY = self.theSceneItemsRect.top()
        self.theSceneLimitsMaxX = self.theSceneItemsRect.right()
        self.theSceneLimitsMaxY = self.theSceneItemsRect.bottom()
       
        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: self.setInputGraphicsScene() real values l,t,r,b, rect = " + \
            str(self.theSceneLimitsMinX)+" "+str(self.theSceneLimitsMinY)+" "+str(self.theSceneLimitsMaxX)+" "+ \
            str(self.theSceneLimitsMaxY)+" "+str(self.theSceneItemsRect)+" ", CDConstants.DebugVerbose )
           
        CDConstants.printOut( "                  itemsBoundingRect itemsBoundingRect itemsBoundingRect itemsBoundingRect = " + \
            str(self.theGraphicsScene.itemsBoundingRect()) , CDConstants.DebugVerbose )

        CDConstants.printOut( "                  sceneRect sceneRect sceneRect sceneRect = " + str(lSceneRect), CDConstants.DebugVerbose )
       

        if (self.cdPreferences is not None):
            # set lTempRectF from dimensions stored in preferences:
            lTempRectF = QtCore.QRectF(0.0, 0.0, \
                self.cdPreferences.pifSceneWidth, self.cdPreferences.pifSceneHeight)
        else:
            # if there are no preferences about the Cell Scene dimensions, take the
            #    **entire scene rect from objects** united with the nominal scene rect:
            lTempRectF = QtCore.QRectF(lSceneRect.united(self.theGraphicsScene.itemsBoundingRect()))

        # set integer-value dimensions (or a QRect, not a QRectF!) when creating a QPixmap:
        lPixmap = QtGui.QPixmap( int(lTempRectF.right() - lTempRectF.left()), \
                                 int(lTempRectF.bottom() - lTempRectF.top()) )
        lPixmap.fill(QtCore.Qt.transparent)

        # prepare a dictionary from new, unique temporary brush colors, to the scene item IDs,
        #   so that we can tell each pixel's item/region just by reading its color
        self.lTmpColorToItemIDDict = dict()

        # prepare a dictionary from new, unique temporary brush colors, to scene item/region keys:
        self.lTmpColorToItemKeyDict = dict()

        # prepare a list of all used unique temporary brush colors:
        self.lTmpColorList = list()

        # deselect all graphics items before rendering the scene to a pixmap:
        self.theGraphicsScene.clearSelection()

        lSceneItems = self.theGraphicsScene.items(QtCore.Qt.AscendingOrder)
        for lItem in lSceneItems:
            #
            # get the scene item/region' key:
            lItemSceneColor = lItem.brush().color().rgba()
            lItemKey = self.colorToKeyRegionDict[lItemSceneColor]
            #
            # store all graphics items' pens and brushes, and assign a unique brush color to each scene item:
            lItem.saveAndClearPen()
            lItem.saveAndClearBrush()
            lItemID = lItem.getRegionID()
            lTmpColor = QtGui.QColor(lItemID)
            lItem.setBrush(QtGui.QBrush(lTmpColor))
            #
            # build a dictionary from new and unique temporary brush colors to scene item IDs:
            self.lTmpColorToItemIDDict[lTmpColor.rgba()] = lItemID
            #
            # build a dictionary from new and unique temporary brush colors to scene item/region keys:
            self.lTmpColorToItemKeyDict[lTmpColor.rgba()] = lItemKey


            # build a list of all the new and unique temporary brush colors:
            if lTmpColor.rgba() in self.lTmpColorList:
                pass
            else:
                self.lTmpColorList.append(lTmpColor.rgba())

            print "CDSceneRasterizer.setInputGraphicsScene() - lItemID =", lItemID
            print "CDSceneRasterizer.setInputGraphicsScene() - lTmpColor.rgba() =", lTmpColor.rgba()
            print "CDSceneRasterizer.setInputGraphicsScene() - self.lTmpColorToItemIDDict[lTmpColor.rgba() =",lTmpColor.rgba(),"] =", self.lTmpColorToItemIDDict[lTmpColor.rgba()]
            print "CDSceneRasterizer.setInputGraphicsScene() - lItem.brush().color().rgba() =", lItem.brush().color().rgba()
        # temporarily disable drawing the scene overlay:
        self.theGraphicsScene.setDrawForegroundEnabled(False)
        print "CDSceneRasterizer.setInputGraphicsScene() - self.lTmpColorToItemIDDict =", self.lTmpColorToItemIDDict
        print "CDSceneRasterizer.setInputGraphicsScene() - self.lTmpColorList =", self.lTmpColorList

        # render the chosen width&height of the scene contents, using a painter into a local pixmap,
        #   with no QPen on any item, and using the new, temporary special brush colors for items:
        lPainter = QtGui.QPainter(lPixmap)
        self.theGraphicsScene.render( lPainter, lSceneRect, lSceneRect, QtCore.Qt.KeepAspectRatio )
        lPainter.end()

        # store the pixmap holding the specially rendered scene:
        self.theRasterizedImageLabel.setPixmap(lPixmap)
        # this QImage is going to hold the rasterized version:
        self.theRasterizedImageLabel.image = lPixmap.toImage()

        self.theRasterizedImageLabel.width = int( lPixmap.width() )
        self.theRasterizedImageLabel.height = int ( lPixmap.height() )
        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: self.setInputGraphicsScene() pGraphicsScene w,h =" + \
              str(self.theRasterizedImageLabel.width) + " " + str(self.theRasterizedImageLabel.height), CDConstants.DebugVerbose )

        # adjusts the size of the label widget to fit its contents (i.e. the pixmap):
        self.theRasterizedImageLabel.adjustSize()
        self.theRasterizedImageLabel.show()
        self.theRasterizedImageLabel.update()


        # restore the original pens and brushes for all graphics items' in the cell and region scene:
        for lItem in lSceneItems:
            lItem.restorePen()
            lItem.restoreBrush()
        # re-enable drawing the scene overlay:
        self.theGraphicsScene.setDrawForegroundEnabled(True)

        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: self.setInputGraphicsScene() done.", CDConstants.DebugExcessive )

    # ------------------------------------------------------------------
    # obtain the rasterized pixmap:
    # ------------------------------------------------------------------
    def getRasterizedPixMap(self):
        lPixmap = self.theRasterizedImageLabel.pixmap()
        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: self.getRasterizedPixMap() done with lPixmap =" + \
            str(lPixmap),  CDConstants.DebugVerbose )
        return lPixmap


    # ------------------------------------------------------------------
    # assign new dict content to the regionsDict global
    # ------------------------------------------------------------------
    def setRegionsDict(self, pDict):
        # clear globals first:
        self.regionsDict = dict()
        self.colorToKeyRegionDict = dict()
        self.colorToNameRegionDict = dict()
        self.colorToCellSizeRegionDict = dict()

        # assign the dictionary of regions/cell types built from found colors:
        self.regionsDict = pDict

        # assign the dictionary of color names (strings from external table):
        lKeys = self.regionsDict.keys()
       
        # build two reverse tables:
        #  colorToKeyRegionDict is used to map color RGBA values
        #     to region keys (integers starting from 1)
        #  colorToNameRegionDict is used to map color RGBA values
        #     to region names (either default strings or user-modified strings)
        #
        for i in xrange(len(self.regionsDict)):
            self.colorToKeyRegionDict[self.regionsDict[lKeys[i]][0].rgba()] = lKeys[i]
            self.colorToNameRegionDict[self.regionsDict[lKeys[i]][0].rgba()] = \
                                       self.regionsDict[lKeys[i]][1]
            self.colorToCellSizeRegionDict[self.regionsDict[lKeys[i]][0].rgba()] = \
                                       self.regionsDict[lKeys[i]][2]
#             print "DIH - DIH ----- CDSceneRasterizer: setRegionsDict()    lKeys[i]  =", lKeys[i]
#             print "DIH - DIH ----- CDSceneRasterizer: setRegionsDict()    self.regionsDict[lKeys[i]][0]  =", self.regionsDict[lKeys[i]][0]
#             print "DIH - DIH ----- CDSceneRasterizer: setRegionsDict()    self.regionsDict[lKeys[i]][0].rgba()  =", self.regionsDict[lKeys[i]][0].rgba()
#             print "DIH - DIH ----- CDSceneRasterizer: setRegionsDict()    self.regionsDict[lKeys[i]][1]  =", self.regionsDict[lKeys[i]][1]
#             print "DIH - DIH ----- CDSceneRasterizer: setRegionsDict()    self.regionsDict[lKeys[i]][2]  =", self.regionsDict[lKeys[i]][2]
#
#         print "___ - DEBUG ----- CDSceneRasterizer: self.setRegionsDict() done with pDict, self.colorToNameRegionDict, self.colorToKeyRegionDict, self.colorToCellSizeRegionDict =", \
#               pDict, self.colorToNameRegionDict, self.colorToKeyRegionDict, self.colorToCellSizeRegionDict










    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # now define fuctions that actually do something with data:
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------










    # ---------------------------------------------------------
    # convert the graphics scene to a fixed-size raster numpy array
    #   this function MUST be called before saving a PIFF file from a scene.
    #   this function only creates an array, which must then converted into PIFF data
    #   by the savePIFFFileFromFixedSizeRaster() function in this same class.
    # ---------------------------------------------------------
    def rasterizeSceneToFixedSizeRaster(self):

        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: rasterizeSceneToFixedSizeRaster() starting.", CDConstants.DebugExcessive )

        # start progress bar:
        self.progressBar.setValue(0)
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        # we use a fixed size raster, so draw a grid on the image label:
        self.theRasterizedImageLabel.drawFixedSizeRaster(True)

        # show the user that the application is busy (while rasterizing the scene):
        # 2011 - Mitja: this doesn't always restore to normal (on different platrforms?) so we don't change cursor for now:
        # QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
   
        lRasterizationSize = self.theRasterizedImageLabel.rasterWidth

        # This used to be set to render the width&height of the scene contents.
        # But we now set what we want as PIFF width & height separately in preferences, so we don't set it thus:
        #    lSceneWidthInPixels = self.theRasterizedImageLabel.width
        #    lSceneHeightInPixels = self.theRasterizedImageLabel.height
        # Instead we take PIFF width & height from the preferences'values as set to the graphics scene,
        #    which we rely on having been assigned to this object's local pointer copy:
        # lSceneWidthInPixels = self.theGraphicsScene.sceneRect().width()
        # lSceneHeightInPixels = self.theGraphicsScene.sceneRect().height()

        lSceneWidthInPixels = self.cdPreferences.pifSceneWidth
        lSceneHeightInPixels = self.cdPreferences.pifSceneHeight


        # now compute how many fixed-sized cells (width and height) there will be in our scene:
        #   use the ceiling function since we may have fractional-sized cells on the edges of the scene
        self.fixedSizeWidthInCells =  int(    math.ceil(  float(lSceneWidthInPixels)  / float(lRasterizationSize)  ) )
        self.fixedSizeHeightInCells = int(    math.ceil(  float(lSceneHeightInPixels) / float(lRasterizationSize)  ) )


        # 2011 - Mitja: create an empty array, into which to write cell values, one for each cell (not one each pixel!) :

        self.fixedSizeCellsArray = numpy.zeros( (self.fixedSizeHeightInCells, self.fixedSizeWidthInCells), dtype=numpy.int )


        # testing, for example this loop fills the first row (y==0) with "1" values:
        #  for i in xrange(0, lSceneWidthInPixels, 1):
        #      # strangely enough, the 1st parameter is rows (y) and the 2nd parameter is columns (x) :
        #      self.fixedSizeCellsArray[0, i] = 1
        #  print self.fixedSizeCellsArray


        k = 1  # 2010 - Mitja: (1.1) keeping track of how many colors have been found

        # set progress bar:
        self.progressBar.setRange(0, (self.fixedSizeHeightInCells * self.fixedSizeWidthInCells) )
        self.progressBar.setValue(0)
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        #self.infoLabel.setText("123456789012345678901234567890123456789012345678901234567890")
        self.infoLabel.setText( "Rasterizing Cell Scene to fixed-size cells (pass 1 of 2) ..." )

        # 2010 - Mitja: python's xrange function is more appropriate for large loops
        #   since it generates integers (the range function generates lists instead)
#         for i in xrange(0, int(lSceneWidthInPixels), lRasterizationSize):
#             for j in xrange(0, int(lSceneHeightInPixels), lRasterizationSize):
        for i in xrange(0, self.fixedSizeWidthInCells, 1):
            # progressBar status update:
            # self.progressBar.setValue(j + (i * self.fixedSizeHeightInCells) )
            self.progressBar.setValue( (i * self.fixedSizeHeightInCells) )
            # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
            #   unless we also explicitly ask Qt to process at least some events.
            QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
            # show the user that the application is busy (while rasterizing the scene):
            # 2011 - Mitja: this doesn't always restore to normal (on different platrforms?) so we don't change cursor for now:
            # QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

            for j in xrange(0, self.fixedSizeHeightInCells, 1):

                xoffset = i * lRasterizationSize
                yoffset = j * lRasterizationSize

                pixelColorList = []
                pixelColorDict = dict()
                maxColorPresence = -1
                prevalentColor = 0

                # print "x,y,sized,list,dict are: ", \
                #   xoffset, yoffset, lRasterizationSize, \
                #   pixelColorList, pixelColorDict

                # find the prevalentColor in the square at xoffset, yoffset:
                for p in xrange(xoffset, xoffset+lRasterizationSize) :
                    for q in xrange(yoffset, yoffset+lRasterizationSize) :

                        # TODO TODO TODO : boundary check on the image size vs. offset+rasterization
                        if (p < lSceneWidthInPixels) and (q < lSceneHeightInPixels) :
                            # obtain the color of the pixel at coordinates p,q as
                            #   QRgb type in the format #AARRGGBB (equivalent to an unsigned int) :

                            ### not from pixmap as in:
                            # lColor = self.theRasterizedImageLabel.originalImage.pixel(p,q)
                            ### but from scene items:

                            # This used to be set to rasterize from width&height of the scene contents.
                            # But we now set what we want as PIFF width & height separately in preferences, so we don't sample it thus:
                            #   lItemAt = self.theGraphicsScene.itemAt(float(p + int(self.theSceneLimitsMinX)),float(q  + int(self.theSceneLimitsMinY)))
                            # Instead we take PIFF width & height from the preferences'values as set to the graphics scene:
                            lItemAt = self.theGraphicsScene.itemAt(float(p),float(q))

                            # convert the current color to an RGBA numeric value:
                            if isinstance( lItemAt, QtGui.QGraphicsItem ):
                                lColor = lItemAt.brush().color().rgba()
                            else:
                                lColor = QtGui.QColor(QtCore.Qt.transparent).rgba()

                            # update the appropriate entry in pixelColorDict
                            try:
                                soFar = pixelColorDict[lColor]
                                pixelColorDict[lColor] = soFar + 1
                            except:
                                pixelColorList.append(lColor)
                                pixelColorDict[lColor] = 1
                            # print "x,y, colors are: ", p, q, pixelColorList, pixelColorDict[lColor]

                # decide the prevalent color for this square of pixels:
                for lColor in pixelColorList:
                    if (pixelColorDict[lColor] > maxColorPresence) :
                        maxColorPresence = pixelColorDict[lColor]
                        prevalentColor = lColor

                # print "x,y,sized,list,dict,max,color are: ", xoffset, yoffset, \
                #   lRasterizationSize, pixelColorList, pixelColorDict, \
                #   maxColorPresence, prevalentColor

                # fill the square at xoffset, yoffset with prevalentColor:

#                 for p in xrange(xoffset, xoffset+lRasterizationSize) :
#                     for q in xrange(yoffset, yoffset+lRasterizationSize) :
#                         # TODO TODO TODO : boundary check on the image size vs. offset+rasterization
#                         if (p < lSceneWidthInPixels) and (q < lSceneHeightInPixels) :
#                             # obtain the color of the pixel at coordinates i,j as
#                             #   QRgb type in the format #AARRGGBB (equivalent to an unsigned int) :
#

                # store the prevalent color in the sampled square of pixels as RGBA numeric value:
                #  (strangely enough, for numpy arrays the 1st parameter is rows (y)
                #       and the 2nd parameter is columns (x)   ) :
                self.fixedSizeCellsArray[j, i] = prevalentColor

                # self.theRasterizedImageLabel.image.setPixel(p,q,prevalentColor)

        # end progress bar:
        self.progressBar.setValue( self.fixedSizeHeightInCells * self.fixedSizeWidthInCells )
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)


        # assign the new image as pixmap:
        #
        # set progress bar:
        self.progressBar.setRange(0, ( lSceneWidthInPixels * lSceneHeightInPixels ) )
        self.progressBar.setValue(0)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        #self.infoLabel.setText("123456789012345678901234567890123456789012345678901234567890")
        self.infoLabel.setText( "Rasterizing Cell Scene to fixed-size cells (pass 2 of 2) ..." )

        for i in xrange(0, int(lSceneWidthInPixels)):
            for j in xrange(0, int(lSceneHeightInPixels)):

                # progressBar status update:
                self.progressBar.setValue(j + (i * lSceneHeightInPixels) )
                # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
                #   unless we also explicitly ask Qt to process at least some events.
                QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)


                # obtain the color of the pixel at coordinates i,j as
                # strangely enough, the 1st parameter is rows (y) and the 2nd parameter is columns (x) :
                p = int ( i / lRasterizationSize)
                q = int ( j / lRasterizationSize)
                self.theRasterizedImageLabel.image.setPixel(i,j,int(self.fixedSizeCellsArray[q, p]))

        self.progressBar.setValue( lSceneWidthInPixels * lSceneHeightInPixels )
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        self.theRasterizedImageLabel.hide()
        self.theRasterizedImageLabel.setPixmap(  \
                QtGui.QPixmap.fromImage(self.theRasterizedImageLabel.image)  )
        self.theRasterizedImageLabel.update()
        self.theRasterizedImageLabel.show()

        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: rasterizeSceneToFixedSizeRaster() done.", CDConstants.DebugExcessive )
        # 2011 - Mitja: this doesn't always restore to normal (on different platrforms?) so we don't change cursor for now:
        # QtGui.QApplication.restoreOverrideCursor()

    # end of def rasterizeSceneToFixedSizeRaster(self)
    # ---------------------------------------------------------



    # ---------------------------------------------------------
    # NOTE: since PIFF assumes RHS coordinates and QImage uses LHS (and so does PNG, etc),
    #       we used to invert Y values here, according to image.height.
    #       But now we take care of the RHS <-> LHS mismatch at its visible end,
    #       by flipping the y coordinate in the QGraphicsView's affine transformations,
    #       as well as immediately when loading a new image or PIFF from file.
    # ---------------------------------------------------------
    def savePIFFFileFromFixedSizeRaster(self, pFileName):

        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: savePIFFFileFromFixedSizeRaster() to " + \
            str(pFileName)+" starting.", CDConstants.DebugExcessive )

        # start progress bar:
        self.progressBar.setValue(0)
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        # we use a fixed size raster, so draw a grid on the image label:
        self.theRasterizedImageLabel.drawFixedSizeRaster(True)

        # open output file, and make sure that it's writable:
        lFile = QtCore.QFile(pFileName)
        lOnlyThePathName,lOnlyTheFileName = os.path.split(str(pFileName))
        if not lFile.open( QtCore.QFile.WriteOnly | QtCore.QFile.Text):
            CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: savePIFFFileFromFixedSizeRaster() cannot write file" + \
                str(pFileName) + " : " + str(lFile.errorString()) + "done.", CDConstants.DebugImportant )
            QtGui.QMessageBox.warning(self, self.tr("CellDraw"),
                    self.tr("Cannot write file %1:\n%2.").arg(pFileName).arg(lFile.errorString()))
            return False
        else:
            #self.infoLabel.setText("123456789012345678901234567890123456789012345678901234567890")
            self.infoLabel.setText( self.tr("Saving fixed-size cells to PIFF file: %1").arg(lOnlyTheFileName) )


        # open a QTextStream, i.e. an "interface for reading and writing text":
        lOutputStream = QtCore.QTextStream(lFile)
        # show the user that the application is busy (while writing to a file):
        # 2011 - Mitja: this doesn't always restore to normal (on different platrforms?) so we don't change cursor for now:
        # QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        lRasterSize = self.theRasterizedImageLabel.rasterWidth
        # lSceneWidthInPixels = self.theRasterizedImageLabel.width
        # lSceneHeightInPixels = self.theRasterizedImageLabel.height
        lSceneWidthInPixels = self.cdPreferences.pifSceneWidth
        lSceneHeightInPixels = self.cdPreferences.pifSceneHeight

        # ------------------------------------------------------------
        # now create a table of total cell type probabilities for each region i.
        #   as set in the self.regionsDict[lRegionsKeys[i]][4][j] dicts for each cell type j:
        lRegionsKeys = self.regionsDict.keys()
        lProbabilityTotalsDict = dict()

        for i in xrange(len(self.regionsDict)):
            for j in xrange( len(self.regionsDict[lRegionsKeys[i]][4]) ):
                # print "at i =", i, ", j =", j, \
                #       "self.regionsDict[keys[i]][4][j] =", \
                #        self.regionsDict[lRegionsKeys[i]][4][j]
                try:
                    soFar = lProbabilityTotalsDict[lRegionsKeys[i]]
                    lProbabilityTotalsDict[lRegionsKeys[i]] = soFar + self.regionsDict[lRegionsKeys[i]][4][j][2]
                except:
                    lProbabilityTotalsDict[lRegionsKeys[i]] = self.regionsDict[lRegionsKeys[i]][4][j][2]
                # print "lProbabilityTotalsDict[lRegionsKeys[i]] = ", \
                #     lProbabilityTotalsDict[lRegionsKeys[i]]

        # ------------------------------------------------------------

        # set progress bar:
        self.progressBar.setRange(0, (self.fixedSizeHeightInCells * self.fixedSizeWidthInCells) )
        self.progressBar.setValue(0)
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        lCellID = 0

#         for x in xrange(0, lSceneWidthInPixels, lRasterSize):
#             for y in xrange(0, lSceneHeightInPixels, lRasterSize):

        # we used to loop through all the pixels in the image label, but we use numpy arrays now:
        for i in xrange(0, self.fixedSizeWidthInCells, 1):
            for j in xrange(0, self.fixedSizeHeightInCells, 1):

                # progressBar status update:
                self.progressBar.setValue(j + (i * self.fixedSizeHeightInCells) )
                # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
                #   unless we also explicitly ask Qt to process at least some events.
                QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

                # retrieve the prevalent color in the fixed-size cell at position i,j :
                #  (strangely enough, for numpy arrays the 1st parameter is rows (j)
                #       and the 2nd parameter is columns (i)   ) :
                lColor = self.fixedSizeCellsArray[j, i]

                if lColor in self.colorToNameRegionDict:

                    # 2010 - Mitja: add functionalities for ignoring white / black regions when saving the PIFF file:
                    if  (self.ignoreWhiteRegionsForPIF == True) and (lColor == QtGui.QColor(QtCore.Qt.white).rgba()):
                        # print ">>>>>>>>>>>>>>>>>>>>>>>> lColor, QtGui.QColor(QtCore.Qt.white) =", lColor, QtGui.QColor(QtCore.Qt.white).rgba()
                        pass # do nothing
                    elif (self.ignoreBlackRegionsForPIF == True) and (lColor == QtGui.QColor(QtCore.Qt.black).rgba()) :
                        # print ">>>>>>>>>>>>>>>>>>>>>>>> lColor, QtGui.QColor(QtCore.Qt.black) =", lColor, QtGui.QColor(QtCore.Qt.black).rgba()
                        pass # do nothing
                    else :

                        lRegionName = self.colorToNameRegionDict[lColor]
   
                        # generate a random floating point number between 0.0 and the region's total probability:
                        lRegionKey = self.colorToKeyRegionDict[lColor]
                        # print " prepare for RANDOM lRegionKey, lColor=self.colorToKeyRegionDict[lColor] =", lRegionKey, lColor
                        lRnd = random.random()
                        # print " RANDOM RANDOM =", lRnd, lRnd * lProbabilityTotalsDict[lRegionKey]
                        lRnd = lRnd * lProbabilityTotalsDict[lRegionKey]
   
                        # loop through all cell type dicts for the current region until probability is matched:
                        lTheCellTypeName = ""
                        lRndCumulative = 0.0
                        for k in xrange(len(  self.regionsDict[lRegionKey][4]  )) :
                            try:
                                lRndCumulative = lRndCumulative + self.regionsDict[lRegionKey][4][k][2]
                                # print "TRY lRndCumulative, self.regionsDict[lRegionKey][4][k][2] =", lRndCumulative, self.regionsDict[lRegionKey][4][k][2]
                            except:
                                lRndCumulative = self.regionsDict[lRegionKey][4][k][2]
                                # print "EXCEPT lRndCumulative, self.regionsDict[lRegionKey][4][k][2] =", lRndCumulative, self.regionsDict[lRegionKey][4][k][2]
                            # if the cell type's probability is matched by the random number, get the cell type name:
                            if ( (lTheCellTypeName == "") and (lRndCumulative > lRnd) ) :
                                lTheCellTypeName = self.regionsDict[lRegionKey][4][k][1]
                                # print "ASSIGN lTheCellTypeName = self.regionsDict[lRegionKey][4][k][1] =", lTheCellTypeName, self.regionsDict[lRegionKey][4][k][1]


                        lX = i * lRasterSize
                        lY = j * lRasterSize
   
                        # todo: to support image scaling, make this work maybe:
                        # xmin = (int)(x*self.scaleXBox)
                        # ymin = (int)(y*self.scaleYBox)
                        xmin = int(lX)
                        ymin = int(lY)
       
                        xmax = xmin+lRasterSize-1
                        if xmax > (lSceneWidthInPixels-1):
                            CDConstants.printOut( "hit X border lSceneWidthInPixels", CDConstants.DebugAll )
                            xmax = lSceneWidthInPixels-1
                        ymax = ymin+lRasterSize-1
                        if ymax > (lSceneHeightInPixels-1):
                            CDConstants.printOut( "hit Y border lSceneHeightInPixels", CDConstants.DebugAll )
                            ymax = lSceneHeightInPixels-1

                        lOutputStream << "%s %s %s %s %s %s 0 0\n"%(lCellID, lTheCellTypeName, xmin, xmax, ymin, ymax)
                        lCellID +=1
                else:
                    # if we caught a color not in the dictionary, we output nothing to the file!
                    # print "GOTCHA! lColor =", lColor, \
                    #     " is not in self.colorToNameRegionDict =", self.colorToNameRegionDict
                    pass

        if (self.savePIFMetadata is True) and (self.cdPreferences is not None):
            lOutputStream << "<xml>\n"
            lOutputStream << "    <units>\n"
            lOutputStream << "        <unit name = \"%s\" />\n" % (str(self.cdPreferences.pifSceneUnits))
            lOutputStream << "    </units>\n"
            lOutputStream << "    <dimensions>\n"
            lOutputStream << "        <width = \"%s\" />\n" % (str(self.cdPreferences.pifSceneWidth))
            lOutputStream << "        <height = \"%s\" />\n" % (str(self.cdPreferences.pifSceneHeight))
            lOutputStream << "        <depth = \"%s\" />\n" % (str(self.cdPreferences.pifSceneDepth))
            lOutputStream << "    </dimensions>\n"
            lOutputStream << "</xml>\n"


        # end progress bar:
        self.progressBar.setValue( self.fixedSizeWidthInCells * self.fixedSizeHeightInCells )
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        # cleanly close access to the file:
        lFile.close()

        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: savePIFFFileFromFixedSizeRaster() PIFF file saving to " + \
            str(pFileName) + " done." , CDConstants.DebugVerbose )
        # 2011 - Mitja: this doesn't always restore to normal (on different platrforms?) so we don't change cursor for now:
        # QtGui.QApplication.restoreOverrideCursor()

    # end of def savePIFFFileFromFixedSizeRaster(self)
    # ---------------------------------------------------------





    # ---------------------------------------------------------
    # convert the graphics scene to a region-raster numpy array.
    #   this function MUST be called before saving a PIFF file from a region-raster scene.
    #   this function only creates an array, which must then converted into PIFF data
    #   by the savePIFFFileFromRegionRasters() function in this same class.
    # ---------------------------------------------------------
    def rasterizeSceneToRegionRasters(self):

        CDConstants.printOut("___ - DEBUG ----- CDSceneRasterizer: rasterizeSceneToRegionRasters() starting.", CDConstants.DebugExcessive )
        #        print "___ - DEBUG ----- CDSceneRasterizer: rasterizeSceneToRegionRasters() starting."

        # start progress bar:
        self.progressBar.setValue(0)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        # we don't use a fixed size raster, so don't draw a grid on the image label:
        self.theRasterizedImageLabel.drawFixedSizeRaster(False)

        # until we get each scene item's raster size, set it to -1 as it's invalid:
        lItemsRasterSizeX = -1
        lItemsRasterSizeY = -1
        lItemsRasterSizeZ = -1

        lSceneWidthInPixels = self.cdPreferences.pifSceneWidth
        lSceneHeightInPixels = self.cdPreferences.pifSceneHeight

        lSceneItems = self.theGraphicsScene.items(QtCore.Qt.AscendingOrder)
        lTotalNumberOfItems = len(lSceneItems)

        # 2011 - Mitja: create two empty arrays, into which to write pixel values, three values for each pixel :
        #     the third dimension is 3 values: region/item (unique) ID is at 0, cell type is at 1, (unique) cellID is at 2
        self.lFinalRegionRasterSceneArray = numpy.zeros( (lSceneHeightInPixels, lSceneWidthInPixels, 3), dtype=numpy.int )
        # the tmp array needs to be replicated for each scene item, hence the 4th dimension == one dataset per item:
        lEachRegionRasterSceneArray = numpy.zeros( (lSceneHeightInPixels, lSceneWidthInPixels, 3, lTotalNumberOfItems), dtype=numpy.int )

        # testing, for example this loop fills the first row (y==0) with "1" values:
        #  for i in xrange(0, lSceneWidthInPixels, 1):
        #      # strangely enough, the 1st parameter is rows (y) and the 2nd parameter is columns (x) :
        #      self.lFinalRegionRasterSceneArray[0, i] = 1
        #  print self.lFinalRegionRasterSceneArray

        # set progress bar:
        self.progressBar.setRange(0, lTotalNumberOfItems)
        self.progressBar.setValue(0)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        #self.infoLabel.setText("123456789012345678901234567890123456789012345678901234567890")
        self.infoLabel.setText( "Rasterizing Cell Scene to region-raster sized cells." )

        lItemCounter = 0
        lCellID = 0
        # build a temporary dictionary from the itemID to the numpy array index counter (the array's 4th dimension) :
        lTmpRegionIDtoItemIndexDict = dict()

        # provide visual feedback to the user, first fill the entire theRasterizedImageLabel pixmap (but not the image!) with transparent color
        lTransparentColor = QtGui.QColor(QtCore.Qt.transparent).rgba()

        # ------------------------------------------------------------
        # now create a table of total cell type probabilities for each region i.
        #   as set in the self.regionsDict[lRegionsKeys[i]][4][j] dicts for each cell type j:
        lRegionsKeys = self.regionsDict.keys()
        lProbabilityTotalsDict = dict()

        for i in xrange(len(self.regionsDict)):
            for j in xrange( len(self.regionsDict[lRegionsKeys[i]][4]) ):
                CDConstants.printOut( "at i = "+str(i)+", j = "+str(j) + \
                      "self.regionsDict[keys[i]][4][j] = "+str(self.regionsDict[lRegionsKeys[i]][4][j]), \
                      CDConstants.DebugAll )
                try:
                    soFar = lProbabilityTotalsDict[lRegionsKeys[i]]
                    lProbabilityTotalsDict[lRegionsKeys[i]] = soFar + self.regionsDict[lRegionsKeys[i]][4][j][2]
                except:
                    lProbabilityTotalsDict[lRegionsKeys[i]] = self.regionsDict[lRegionsKeys[i]][4][j][2]
                CDConstants.printOut("lProbabilityTotalsDict[lRegionsKeys[i]] = " + \
                    str(lProbabilityTotalsDict[lRegionsKeys[i]]), CDConstants.DebugAll )

        # ------------------------------------------------------------

        for lItem in lSceneItems:

            # map from scene correctly, i.e. using inverse scene transform
            #   for translation/rotation/scaling of the object:
            # print "lItem.sceneTransform() = ", lItem.sceneTransform()
            # print "lItem.sceneTransform().inverted() = ", lItem.sceneTransform().inverted()
            # NOTE: in PyQt, inverted() returns a tuple:
            lItemInverseTransform,lIsNotSingular = lItem.sceneTransform().inverted()
            if lIsNotSingular is False:
                QtGui.QMessageBox.warning(self, "CellDraw", \
                    self.tr("Can't use scene item %1: singular QTransform.").arg(lItem))
                return False

            lItemSceneColor = lItem.brush().color().rgba()

            if lItemSceneColor in self.colorToNameRegionDict:
                lItemsRasterSizeList = self.colorToCellSizeRegionDict[lItemSceneColor]
            else:
                QtGui.QMessageBox.warning(self, "CellDraw", \
                    self.tr("Can't use scene item %1:\nMost likely a white (\"hole\") region,\nwhich is currently unsupported for PIFs with region-rasters.\nYou can remove white regions, or save as PIFF with fixed raster size.").arg(lItem.toolTip()))
                return False

            # 2011 - Mitja: first step towards 3D PIFF scenes:
            #   add a 3rd dimension to cell sizes (one layer only, alas).

            # rasterize "region items" differently from "cell items" in the Cell Scene!!!   :
            if (lItem.itsaRegionOrCell == CDConstants.ItsaRegionConst) :
                lItemsRasterSizeX = int(lItemsRasterSizeList[0])
                lItemsRasterSizeY = int(lItemsRasterSizeList[1])
                lItemsRasterSizeZ = int(lItemsRasterSizeList[2])
            else:
                #  ( TODO: change these values to MAX_INT or something similar: )
                lItemsRasterSizeX = 9999
                lItemsRasterSizeY = 9999
                lItemsRasterSizeZ = 9999

            CDConstants.printOut( "lItemsRasterSizeX,Y,Z = "+str(lItemsRasterSizeX)+ \
                    " "+str(lItemsRasterSizeY)+" "+str(lItemsRasterSizeZ), CDConstants.DebugVerbose )

            lRegionKey = self.colorToKeyRegionDict[lItemSceneColor]
            lItemID = lItem.getRegionID()
            # fill the temporary dictionary from the itemID to the numpy array index counter (the array's 4 dimension) :
            lTmpRegionIDtoItemIndexDict[lItemID] = lItemCounter

            # tmp array fill part A - fill the entire array with rectangular cells:

            # grab the boundingRect from the item's polygon: the rectangle is in items' local coordinates:
            lItemPolygonBoundingRect = lItem.polygon().boundingRect()
            lItemPolygonTransformedTopLeft = lItem.sceneTransform().map(lItemPolygonBoundingRect.topLeft())
            lItemPolygonTransformedBottomRight = lItem.sceneTransform().map(lItemPolygonBoundingRect.bottomRight())
            lItemPolygonTransformedRect = QtCore.QRectF(lItemPolygonTransformedTopLeft, lItemPolygonTransformedBottomRight)
            CDConstants.printOut( "rasterizeVarSizedCellRegionsAndSavePIF - lItemPolygonBoundingRect = "+ \
                str(lItemPolygonBoundingRect)+" lItemPolygonTransformedRect = "+str(lItemPolygonTransformedRect), CDConstants.DebugAll )

            # provide user feedback, draw an outline of the current item's bounding rectangle:
            lTmpColor = QtGui.QColor(QtCore.Qt.black).rgba()
            xmin = int(lItemPolygonTransformedTopLeft.x())
            ymin = int(lItemPolygonTransformedTopLeft.y())
            xmax = int(lItemPolygonTransformedBottomRight.x())
            ymax = int(lItemPolygonTransformedBottomRight.y())
            self.theRasterizedImageLabel.plotRect(lTmpColor, xmin, ymin, xmax, ymax)
            self.theRasterizedImageLabel.update()

            # progressBar status update:
            self.progressBar.setValue(lItemCounter)
            QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

            for i in xrange(xmin, xmax, lItemsRasterSizeX):
                for j in xrange(ymin, ymax, lItemsRasterSizeY):

#             for i in xrange(0, lSceneWidthInPixels, lItemsRasterSizeX):
#                 for j in xrange(0, lSceneHeightInPixels, lItemsRasterSizeY):

#                     # find the point at lItem's local coordinates that corresponds to i,j at scene coordinates:
#                     lLocalToItemPointF = lItemInverseTransform.map( QtCore.QPointF(float(i),float(j)) )
#                     # check if lItem contains the (i,j)-to-local point:
#                     if lItem.contains( lLocalToItemPointF ):
#                         # now store the numpy array's points that belongs to the current item:
#                         lItemSceneColor = lItem.brush().color().rgba()
#                         lRegionKey = self.colorToKeyRegionDict[lItemSceneColor]
#                         lEachRegionRasterSceneArray[j, i, 0, lItemID] = lItemSceneColor

                    xoffset = i
                    yoffset = j

                    # ------------------------------------------------------------
                    # find the cell type (both name and key) according to region probabilities:
                    #
                    # generate a random floating point number between 0.0 and the region's total probability:
                    lTmpColor = lItem.brush().color().rgba()
                    lRegionKey = self.colorToKeyRegionDict[lTmpColor]
                    lRnd = random.random()
                    lRnd = lRnd * lProbabilityTotalsDict[lRegionKey]

                    # loop through all cell type dicts for the current region until probability is matched:
                    lTheCellTypeName = ""
                    lTheCellTypeKey = -1
                    lRndCumulative = 0.0
                    for k in xrange(len(  self.regionsDict[lRegionKey][4]  )) :
                        try:
                            lRndCumulative = lRndCumulative + self.regionsDict[lRegionKey][4][k][2]
                        except:
                            lRndCumulative = self.regionsDict[lRegionKey][4][k][2]
                        # if the cell type's probability is matched by the random number, get the cell type name:
                        if ( (lTheCellTypeName == "") and (lRndCumulative > lRnd) ) :
                            lTheCellTypeName = self.regionsDict[lRegionKey][4][k][1]
                            lTheCellTypeKey = k
                    # ------------------------------------------------------------

# 
#                     # to provide user feedback, generate a random color for the current lCellID:
#                     lTmpRgbaColor = QtGui.QColor( int(random.random()*256.0), \
#                                                   int(random.random()*256.0), \
#                                                   int(random.random()*256.0) ).rgba()
#                     lTmpColor = QtGui.QColor(lTmpRgbaColor)
                    # to provide user feedback, get the color for the current lCellID from the regions table:
                    lTmpColor = QtGui.QColor(self.regionsDict[lRegionKey][4][lTheCellTypeKey][0])
                    lPen = QtGui.QPen(lTmpColor)
                    lPen.setCosmetic(True) # cosmetic pen = width always 1 pixel wide, independent of painter's transformation set
                    lPainter = QtGui.QPainter(self.theRasterizedImageLabel.pixmap())
                    lPainter.setPen(lPen)
                    # CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: rasterizeSceneToRegionRasters() lTmpColor = "+str(lTmpColor), CDConstants.DebugExcessive )

                    # fill a regionID, cellID value in the square at xoffset, yoffset:
                    for p in xrange(xoffset, xoffset+lItemsRasterSizeX, 1) :
                        for q in xrange(yoffset, yoffset+lItemsRasterSizeY, 1) :
                            if (p < lSceneWidthInPixels) and (q < lSceneHeightInPixels) and (p >= 0) and (q >= 0):
                                lEachRegionRasterSceneArray[p, q, 0, lItemCounter] = lItemID
                                lEachRegionRasterSceneArray[p, q, 1, lItemCounter] = lTheCellTypeKey
                                lEachRegionRasterSceneArray[p, q, 2, lItemCounter] = lCellID
                                # set the pixmap to provide user feedback:
                                lPainter.drawPoint(p, q)
                    lPainter.end()
                    self.theRasterizedImageLabel.update()

                    # the cell's possible pixels have all been filled, increment lCellID:
                    lCellID = lCellID + 1

            # end of for i - for j tmp array fill

            lItemCounter = lItemCounter + 1
            # end progress bar:
            self.progressBar.setValue( lSceneHeightInPixels * lSceneWidthInPixels )
            QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        # end of "for lItem in lSceneItems" AKA part A

        # part B, copy data from tmp numpy arrays to final array using the pixmap/image,
        #    so that we can save to PIFF in the end!

        lCellID = 0
        lTmpRgbaColorDict = dict()

        # set progress bar:
        self.progressBar.setRange(0, ( lSceneWidthInPixels * lSceneHeightInPixels ) )
        self.progressBar.setValue(0)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        for i in xrange(0, lSceneWidthInPixels, 1):

            # provide feedback to the user:
            self.theRasterizedImageLabel.setPixmap(  \
                    QtGui.QPixmap.fromImage(self.theRasterizedImageLabel.image)  )
            self.theRasterizedImageLabel.update()

            for j in xrange(0, lSceneHeightInPixels, 1):
                # get the QRgb value at coordinates (i,j) in the rendered image from the scene,
                #   since that color will identify the region type at (i,j) coordinates:
                lPixelColor = self.theRasterizedImageLabel.image.pixel(i, j)
                # print "lPixelColor = ", lPixelColor
                if lPixelColor in self.lTmpColorList:
                    # find the unique RegionID for the current pixel:
                    lRegionIDatPixel = self.lTmpColorToItemIDDict[lPixelColor]
                    # retrieve the index value for the RegionID, to access data in numpy arrays:
                    lItemIndex = lTmpRegionIDtoItemIndexDict[lRegionIDatPixel]
                    # finally get the cell type key and the unique CellID for the pixel:
                    lTheCellTypeKey = lEachRegionRasterSceneArray[i, j, 1, lItemIndex]
                    lCellID = lEachRegionRasterSceneArray[i, j, 2, lItemIndex]

                    # set final array content:
                    self.lFinalRegionRasterSceneArray[j, i, 0] = lRegionIDatPixel
                    self.lFinalRegionRasterSceneArray[j, i, 1] = lTheCellTypeKey
                    self.lFinalRegionRasterSceneArray[j, i, 2] = lCellID

                    # to provide user feedback, generate a random color for
                    #   the current lTheCellTypeKey and region:
                    lTmpColorfulCellTypeID = lTheCellTypeKey + 1000 * lItemIndex
                    if lTmpColorfulCellTypeID in lTmpRgbaColorDict:
                        lTmpRgbaColor = lTmpRgbaColorDict[lTmpColorfulCellTypeID]
                    else:
#                         lRed = int(lCellID % 256)
#                         lGreen = int(   int((lCellID - lRed) / 256)  % 256)
#                         lBlue = int(   int((lCellID - lGreen) / 256)  % 256)
#                         lTmpRgbaColor = QtGui.QColor( lRed, \
#                                                       lGreen, \
#                                                       lBlue ).rgba()
                        lTmpRgbaColor = QtGui.QColor( int(random.random()*256.0), \
                                                      int(random.random()*256.0), \
                                                      int(random.random()*256.0) ).rgba()
                        lTmpRgbaColorDict[lTmpColorfulCellTypeID] = lTmpRgbaColor

                    lRegionKey = self.lTmpColorToItemKeyDict[lPixelColor]
                    # to provide user feedback, get the color for the current lCellID from the regions table:
                    lTmpRgbaColor = QtGui.QColor(self.regionsDict[lRegionKey][4][lTheCellTypeKey][0]).rgba()
                    self.theRasterizedImageLabel.image.setPixel(i,j,lTmpRgbaColor)
                else:
                    # transparent color means no item at these pixel coordinates:
                    self.theRasterizedImageLabel.image.setPixel(i,j,lTransparentColor)
                    # clear the values at the current pixel coords in the finaly numpy array,
                    #    since this point does not belong to any item in the scene:
                    self.lFinalRegionRasterSceneArray[j, i, 0] = -1
                    self.lFinalRegionRasterSceneArray[j, i, 1] = -1
                    self.lFinalRegionRasterSceneArray[j, i, 2] = -1

                # progressBar status update:
                self.progressBar.setValue(j + (i * lSceneHeightInPixels) )
                QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        # end progress bar:
        self.progressBar.setValue( lSceneHeightInPixels * lSceneWidthInPixels )
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)


        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: rasterizeSceneToRegionRasters() copy from numpy array to image." , CDConstants.DebugExcessive )
        self.theRasterizedImageLabel.setPixmap(  \
                QtGui.QPixmap.fromImage(self.theRasterizedImageLabel.image)  )
        self.theRasterizedImageLabel.update()


        # provide user feedback, draw an outline of all the items' bounding rectangles:
        lTmpColor = QtGui.QColor(QtCore.Qt.black).rgba()
        for lItem in lSceneItems:
            # grab the boundingRect from each item's polygon -- the rectangle is in item's local coordinates:
            lItemPolygonBoundingRect = lItem.polygon().boundingRect()
            lItemPolygonTransformedTopLeft = lItem.sceneTransform().map(lItemPolygonBoundingRect.topLeft())
            lItemPolygonTransformedBottomRight = lItem.sceneTransform().map(lItemPolygonBoundingRect.bottomRight())
            lItemPolygonTransformedRect = QtCore.QRectF(lItemPolygonTransformedTopLeft, lItemPolygonTransformedBottomRight)
            CDConstants.printOut( "rasterizeVarSizedCellRegionsAndSavePIF - lItemPolygonBoundingRect = " + \
                str(lItemPolygonBoundingRect) + " lItemPolygonTransformedRect = " + str(lItemPolygonTransformedRect), CDConstants.DebugAll )
            xmin = lItemPolygonTransformedTopLeft.x()
            ymin = lItemPolygonTransformedTopLeft.y()
            xmax = lItemPolygonTransformedBottomRight.x()
            ymax = lItemPolygonTransformedBottomRight.y()
            self.theRasterizedImageLabel.plotRect(lTmpColor, xmin, ymin, xmax, ymax)
        self.theRasterizedImageLabel.update()


        # 2011 - Mitja: empty the numpy arrays, just in case, to free memory taken by numpy:
        lEachRegionRasterSceneArray = numpy.zeros(1)

        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: rasterizeSceneToRegionRasters() done.", CDConstants.DebugExcessive )

    # end of def rasterizeSceneToRegionRasters(self)
    # ---------------------------------------------------------


    # ---------------------------------------------------------
    # NOTE: since PIFF assumes RHS coordinates and QImage uses LHS (and so does PNG, etc),
    #       we used to invert Y values here, according to image.height.
    #       But now we take care of the RHS <-> LHS mismatch at its visible end,
    #       by flipping the y coordinate in the QGraphicsView's affine transformations,
    #       as well as immediately when loading a new image or PIFF from file.
    # ---------------------------------------------------------
    def savePIFFFileFromRegionRasters(self, pFileName):

        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: savePIFFFileFromRegionRasters() to" + \
            str(pFileName) + " starting.", CDConstants.DebugExcessive )

        # open output file, and make sure that it's writable:
        lFile = QtCore.QFile(pFileName)
        lOnlyThePathName,lOnlyTheFileName = os.path.split(str(pFileName))
        if not lFile.open( QtCore.QFile.WriteOnly | QtCore.QFile.Text):
            CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: savePIFFFileFromRegionRasters() cannot write file" + \
                str(pFileName) + " : " + str(lFile.errorString()) + "done.", CDConstants.DebugImportant )
            QtGui.QMessageBox.warning(self, self.tr("CellDraw"),
                    self.tr("Cannot write file %1:\n%2.").arg(pFileName).arg(lFile.errorString()))
            return False
        else:
            #self.infoLabel.setText("123456789012345678901234567890123456789012345678901234567890")
            self.infoLabel.setText( self.tr("Saving region-raster cells to PIFF file: %1").arg(lOnlyTheFileName) )

        # open a QTextStream, i.e. an "interface for reading and writing text":
        lOutputStream = QtCore.QTextStream(lFile)

        lSceneItems = self.theGraphicsScene.items(QtCore.Qt.AscendingOrder)
        lTotalNumberOfItems = len(lSceneItems)

        lSceneWidthInPixels = self.cdPreferences.pifSceneWidth
        lSceneHeightInPixels = self.cdPreferences.pifSceneHeight
        lSceneDepthInPixels = self.cdPreferences.pifSceneDepth

        # set progress bar:
        self.progressBar.setRange(0, (lSceneHeightInPixels * lSceneWidthInPixels) )
        self.progressBar.setValue(0)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        # we used to loop through all the pixels in the image label, but we use numpy arrays now:
        for i in xrange(0, lSceneWidthInPixels, 1):
            for j in xrange(0, lSceneHeightInPixels, 1):

                # progressBar status update:
                self.progressBar.setValue(j + (i * lSceneHeightInPixels) )
                QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

                # get final array content:
                lRegionIDatPixel = self.lFinalRegionRasterSceneArray[j, i, 0]
                lTheCellTypeKey = self.lFinalRegionRasterSceneArray[j, i, 1]
                lCellID = self.lFinalRegionRasterSceneArray[j, i, 2]
                if (lRegionIDatPixel >= 0) and (lTheCellTypeKey >= 0) and (lCellID >= 0) :

                    # find the region color for the current pixel's region
                    for lItem in lSceneItems:
                        if (lItem.getRegionID() == lRegionIDatPixel):
                            lItemSceneColor = lItem.brush().color().rgba()
                    # find the region key for the current pixel's region color:
                    lRegionKey = self.colorToKeyRegionDict[lItemSceneColor]
                   
                    # finally retrieve the cell name string for the current pixel's cell type:
                    lTheCellTypeName = self.regionsDict[lRegionKey][4][lTheCellTypeKey][1]

                    # 2011 - Mitja: first step towards 3D PIFF scenes:
                    #   find the cell region's raster size in the z dimension:
                    lItemsRasterSizeList = self.colorToCellSizeRegionDict[lItemSceneColor]
                    lItemsRasterSizeZ = lItemsRasterSizeList[2]

                    xmin = i
                    xmax = i
                    ymin = j
                    ymax = j
                    zmin = 0
                    zmax = zmin+lItemsRasterSizeZ-1
                    # check that we don't go out of bounds in the z dimension:
                    if (zmax >= lSceneDepthInPixels):
                        zmax = lSceneDepthInPixels-1
                    if (zmax < 0):
                        zmax = 0
                    lOutputStream << "%s %s %s %s %s %s %s %s\n"%(lCellID, lTheCellTypeName, xmin, xmax, ymin, ymax, zmin, zmax)
                else:
                    pass

        if (self.savePIFMetadata is True) and (self.cdPreferences is not None):
            lOutputStream << "<xml>\n"
            lOutputStream << "    <units>\n"
            lOutputStream << "        <unit name = \"%s\" />\n" % (str(self.cdPreferences.pifSceneUnits))
            lOutputStream << "    </units>\n"
            lOutputStream << "    <dimensions>\n"
            lOutputStream << "        <width = \"%s\" />\n" % (str(self.cdPreferences.pifSceneWidth))
            lOutputStream << "        <height = \"%s\" />\n" % (str(self.cdPreferences.pifSceneHeight))
            lOutputStream << "        <depth = \"%s\" />\n" % (str(self.cdPreferences.pifSceneDepth))
            lOutputStream << "    </dimensions>\n"
            lOutputStream << "</xml>\n"

        # end progress bar:
        self.progressBar.setValue(lSceneHeightInPixels * lSceneWidthInPixels)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        # cleanly close access to the file:
        lFile.close()

        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: savePIFFFileFromRegionRasters() PIFF file saving to" + \
            str(pFileName) + "done.", CDConstants.DebugAll )

    # end of def savePIFFFileFromRegionRasters(self)
    # ---------------------------------------------------------






    # ---------------------------------------------------------
    # convert the graphics items (cell regions) to a rasterized image
    #   this function also saves a PIFF file from a scene
    # ---------------------------------------------------------
    def rasterizeVarSizedCellRegionsAndSavePIF(self, pFileName):

        # start progress bar:
        self.progressBar.setValue(0)
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        # we don't use a fixed size raster, so don't draw a grid on the image label:
        self.theRasterizedImageLabel.drawFixedSizeRaster(False)

        # open output file, and make sure that it's writable:
        lFile = QtCore.QFile(pFileName)
        lOnlyThePathName,lOnlyTheFileName = os.path.split(str(pFileName))
        if not lFile.open( QtCore.QFile.WriteOnly | QtCore.QFile.Text):
            QtGui.QMessageBox.warning(self, "CellDraw", \
                    self.tr("Cannot write file %1 .\nError: [%2] .").arg(lOnlyTheFileName).arg(lFile.errorString()))
            return False
        else:
            #self.infoLabel.setText("123456789012345678901234567890123456789012345678901234567890")
            self.infoLabel.setText( self.tr("Rasterizing region-raster cells, saving to PIFF file: %1").arg(lOnlyTheFileName) )

        # open a QTextStream, i.e. an "interface for reading and writing text":
        lOutputStream = QtCore.QTextStream(lFile)
        # show the user that the application is busy (while writing to a file):
        # 2011 - Mitja: setOverrideCursor doesn't always restore to normal (on some platrforms?) so we don't change cursor for now:
        # QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)


        # This used to be set to render the width&height of the scene contents.
        # But we now set what we want as PIFF width & height separately in preferences:
        lSceneWidthInPixels = self.cdPreferences.pifSceneWidth
        lSceneHeightInPixels = self.cdPreferences.pifSceneHeight
        lSceneRect = QtCore.QRectF(0, 0, lSceneWidthInPixels, lSceneHeightInPixels)

        # ------------------------------------------------------------
        # now create a table of total cell type probabilities for each region i.
        #   as set in the self.regionsDict[lRegionsKeys[i]][4][j] dicts for each cell type j:
        lRegionsKeys = self.regionsDict.keys()
        lProbabilityTotalsDict = dict()

        for i in xrange(len(self.regionsDict)):
            for j in xrange( len(self.regionsDict[lRegionsKeys[i]][4]) ):
                # print "at i =", i, ", j =", j, \
                #       "self.regionsDict[keys[i]][4][j] =", \
                #        self.regionsDict[lRegionsKeys[i]][4][j]
                try:
                    soFar = lProbabilityTotalsDict[lRegionsKeys[i]]
                    lProbabilityTotalsDict[lRegionsKeys[i]] = soFar + self.regionsDict[lRegionsKeys[i]][4][j][2]
                except:
                    lProbabilityTotalsDict[lRegionsKeys[i]] = self.regionsDict[lRegionsKeys[i]][4][j][2]
                # print "lProbabilityTotalsDict[lRegionsKeys[i]] = ", \
                #     lProbabilityTotalsDict[lRegionsKeys[i]]
        # ------------------------------------------------------------


        lSceneItems = self.theGraphicsScene.items(QtCore.Qt.AscendingOrder)
        lTotalNumberOfItems = len(lSceneItems)

        # set progress bar:
        self.progressBar.setRange(0, lTotalNumberOfItems)
        self.progressBar.setValue(0)
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        lItemCounter = 0

        CDConstants.printOut( "", CDConstants.DebugExcessive )
        CDConstants.printOut( "rasterizeVarSizedCellRegionsAndSavePIF - running....", CDConstants.DebugExcessive )
        self.infoLabel.setText("Rasterizing PIFF elements from region rasters ...")
       
        # saving PIFF file cell IDs starting from 0 onwards:
        lCellID = 0
       
        for lItem in lSceneItems:
       
            lItemCounter = lItemCounter + 1

            # progressBar status update:
            self.progressBar.setValue(lItemCounter)
            # Qt/PyQt's progressBar won't display updates from setValue() calls,
            #   unless we also explicitly ask Qt to process at least some events.
            QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

            lColor = lItem.brush().color().rgba()

            if lColor in self.colorToNameRegionDict:
                lItemsRasterSizeList = self.colorToCellSizeRegionDict[lColor]
            else:
                QtGui.QMessageBox.warning(self, "CellDraw", \
                    self.tr("Can't use scene item %1:\nMost likely a white (\"hole\") region,\nwhich is currently unsupported for PIFs with region-rasters.\nYou can remove white regions, or save as PIFF with fixed raster size.").arg(lItem.toolTip()))
                return False
           
            # map from scene correctly, i.e. using inverse scene transform
            #   for translation/rotation/scaling of the object:
            # print "lItem.sceneTransform() = ", lItem.sceneTransform()
            # print "lItem.sceneTransform().inverted() = ", lItem.sceneTransform().inverted()
            # NOTE: in PyQt, inverted() returns a tuple:
            lItemInverseTransform,lIsNotSingular = lItem.sceneTransform().inverted()
            if lIsNotSingular is False:
                QtGui.QMessageBox.warning(self, "CellDraw", \
                    self.tr("Can't use scene item %1: singular QTransform.").arg(lItem))
                return False


            # 2011 - Mitja: first step towards 3D PIFF scenes:
            #   add a 3rd dimension to cell sizes (one layer only, alas).


            # rasterize "region items" differently from "cell items" in the Cell Scene!!!   :
            if (lItem.itsaRegionOrCell == CDConstants.ItsaRegionConst) :
                lItemsRasterSizeX = int(lItemsRasterSizeList[0])
                lItemsRasterSizeY = int(lItemsRasterSizeList[1])
                lItemsRasterSizeZ = int(lItemsRasterSizeList[2])
            else:
                #  ( TODO: change these values to MAX_INT or something similar: )
                lItemsRasterSizeX = 9999
                lItemsRasterSizeY = 9999
                lItemsRasterSizeZ = 9999

            CDConstants.printOut( "lItemsRasterSizeX,Y,Z = " + \
                str(lItemsRasterSizeX) +" "+ str(lItemsRasterSizeY) +" "+ str(lItemsRasterSizeZ), CDConstants.DebugVerbose )

            # start from item's 0,0 and not from SCENE's 0,0:
            # 1. first find where in the scene is this item's 0,0, then go from there.

            # grab the boundingRect from the item: the rectangle is in items' local coordinates:
            #
            #  the QGraphicsItem's boundingRect is WRONG for us, since it includes rendering artifacts such as the item's pen!
            #
            # lItemBoundingRect = lItem.boundingRect()
            # lItemTransformedTopLeft = lItem.sceneTransform().map(lItemBoundingRect.topLeft())
            # lItemTransformedBottomRight = lItem.sceneTransform().map(lItemBoundingRect.bottomRight())
            # lItemTransformedRect = QtCore.QRectF(lItemTransformedTopLeft, lItemTransformedBottomRight)
            # print "rasterizeVarSizedCellRegionsAndSavePIF - lItemBoundingRect =", lItemBoundingRect, ", lItemTransformedRect =", lItemTransformedRect

            # grab the boundingRect from the item's polygon: the rectangle is in items' local coordinates:
            lItemPolygonBoundingRect = lItem.polygon().boundingRect()
            lItemPolygonTransformedTopLeft = lItem.sceneTransform().map(lItemPolygonBoundingRect.topLeft())
            lItemPolygonTransformedBottomRight = lItem.sceneTransform().map(lItemPolygonBoundingRect.bottomRight())
            lItemPolygonTransformedRect = QtCore.QRectF(lItemPolygonTransformedTopLeft, lItemPolygonTransformedBottomRight)
            CDConstants.printOut( "rasterizeVarSizedCellRegionsAndSavePIF - lItemPolygonBoundingRect = " + \
                str(lItemPolygonBoundingRect) + " , lItemPolygonTransformedRect = " + str(lItemPolygonTransformedRect), CDConstants.DebugExcessive )

            # provide user feedback, draw an outline of the current item's bounding rectangle:
            lTmpColor = QtGui.QColor(QtCore.Qt.black).rgba()
            xmin = lItemPolygonTransformedTopLeft.x()
            ymin = lItemPolygonTransformedTopLeft.y()
            xmax = lItemPolygonTransformedBottomRight.x()
            ymax = lItemPolygonTransformedBottomRight.y()
            self.theRasterizedImageLabel.plotRect(lTmpColor, xmin, ymin, xmax, ymax)
            self.theRasterizedImageLabel.update()


#
# TODO TODO TODO:
#
# change this rasterization to be of the same type as for fixed, pass through ARRAY!
#


            for i in xrange(lItemPolygonTransformedTopLeft.x(), lItemPolygonTransformedBottomRight.x(), lItemsRasterSizeX):
                for j in xrange(lItemPolygonTransformedTopLeft.y(), lItemPolygonTransformedBottomRight.y(), lItemsRasterSizeY):
   
                    #   map from scene correctly, i.e. using the inverse scene transform
                    #   for translation/rotation/scaling of the object:
                   
                    lCellTopLeftPoint = lItemInverseTransform.map( QtCore.QPointF(i,j) )
                    lCellBottomRightPoint = lItemInverseTransform.map( \
                        QtCore.QPointF( \
                            float(i+lItemsRasterSizeX-1), \
                            float(j+lItemsRasterSizeY-1)    )     )
                    CDConstants.printOut( "i , j, lItemsRasterSizeX, lItemsRasterSizeY, lCellTopLeftPoint, lCellBottomRightPoint = " + \
                        str(i)+" "+str(j)+" "+str(lItemsRasterSizeX)+" "+str(lItemsRasterSizeY)+" "+str(lCellTopLeftPoint)+" "+ \
                        str(lCellBottomRightPoint), CDConstants.DebugAll )

                    # region AND scene inside/outside test for this cell's starting and ending points:
                    if (lItem.contains( lCellTopLeftPoint ) or lItem.contains( lCellBottomRightPoint )) \
                        and lSceneRect.contains( lCellTopLeftPoint ) and lSceneRect.contains( lCellBottomRightPoint ):
                        # inside:
                        CDConstants.printOut( "endpoint INSIDE", CDConstants.DebugSparse )
                        lColor = lItem.brush().color().rgba()
                       
                    else:
                        # outside:
                        CDConstants.printOut( "endpoint OUTSIDE", CDConstants.DebugSparse )
                        lColor = QtGui.QColor(QtCore.Qt.transparent).rgba()

                    if lColor in self.colorToNameRegionDict:
                        lRegionName = self.colorToNameRegionDict[lColor]
                        # please note: i,j are integers because they're produced by Python's xrange, which can only deal with integers. boooo!
                        xmin = i
                        xmax = xmin+lItemsRasterSizeX-1
                        ymin = j
                        ymax = ymin+lItemsRasterSizeY-1
                        # 2011 - Mitja: first step towards 3D PIFF scenes:
                        zmin = 0
                        zmax = zmin+lItemsRasterSizeZ-1

#                         xmax = xmin+lItemsRasterSizeX-1
#                         if xmax > (lSceneWidthInPixels-1):
#                             print "hit X border lSceneWidthInPixels"
#                             xmax = lSceneWidthInPixels-1
#                         ymax = ymin+lItemsRasterSizeY-1
#                         if ymax > (lSceneHeightInPixels-1):
#                             print "hit Y border lSceneHeightInPixels"
#                             ymax = lSceneHeightInPixels-1
#                         zmax = zmin+lItemsRasterSizeZ-1
#                         if zmax < zmin:
#                             print "3D PIFF scene - just hit Z dimension snag: zmin=",zmin,"zmax=",zmax
#                             zmax = zmin
#                             print "3D PIFF scene - correcting...      ...now: zmin=",zmin,"zmax=",zmax

                        # Finally generate one PIFF line to describe the current cell:
                        # ------------------------------------------------------------
                        # generate a random floating point number between 0.0 and the region's total probability:
                        lRegionKey = self.colorToKeyRegionDict[lColor]
                        lRnd = random.random()
                        lRnd = lRnd * lProbabilityTotalsDict[lRegionKey]

                        # loop through all cell type dicts for the current region until probability is matched:
                        lTheCellTypeName = ""
                        lRndCumulative = 0.0
                        for k in xrange(len(  self.regionsDict[lRegionKey][4]  )) :
                            try:
                                lRndCumulative = lRndCumulative + self.regionsDict[lRegionKey][4][k][2]
                            except:
                                lRndCumulative = self.regionsDict[lRegionKey][4][k][2]
                            # if the cell type's probability is matched by the random number, get the cell type name:
                            if ( (lTheCellTypeName == "") and (lRndCumulative > lRnd) ) :
                                lTheCellTypeName = self.regionsDict[lRegionKey][4][k][1]
                        # ------------------------------------------------------------

                        # provide user feedback:
                        self.theRasterizedImageLabel.plotRect(lColor, xmin, ymin, xmax, ymax)
                        #
                        self.theRasterizedImageLabel.update()
                        lOutputStream << "%s %s %s %s %s %s %s %s\n"%(lCellID, lTheCellTypeName, xmin, xmax, ymin, ymax, zmin, zmax)
                        lCellID +=1

                    else:
                        # if we caught a color not in the dictionary, we output nothing to the file!
                        pass


        if (self.savePIFMetadata is True) and (self.cdPreferences is not None):
            lOutputStream << "<xml>\n"
            lOutputStream << "    <units>\n"
            lOutputStream << "        <unit name = \"%s\" />\n" % (str(self.cdPreferences.pifSceneUnits))
            lOutputStream << "    </units>\n"
            lOutputStream << "    <dimensions>\n"
            lOutputStream << "        <width = \"%s\" />\n" % (str(self.cdPreferences.pifSceneWidth))
            lOutputStream << "        <height = \"%s\" />\n" % (str(self.cdPreferences.pifSceneHeight))
            lOutputStream << "    </dimensions>\n"
            lOutputStream << "</xml>\n"


        # end progress bar:
        self.progressBar.setValue( lTotalNumberOfItems )
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        # cleanly close access to the file:
        lFile.close()

        #self.infoLabel.setText("123456789012345678901234567890123456789012345678901234567890")
        self.infoLabel.setText( self.tr("Saving region-raster cells to PIFF file %1 now complete.").arg(lOnlyTheFileName) )

        CDConstants.printOut( "rasterizeVarSizedCellRegionsAndSavePIF - ....PIFF file rasterization from individual items complete.\n" , CDConstants.DebugExcessive )


    # end of def rasterizeVarSizedCellRegionsAndSavePIF(self)
    # ---------------------------------------------------------






    # ---------------------------------------------------------
    # convert the graphics items (cell regions) to a rasterized image
    #   this function also saves a PIFF file from a scene
    # ---------------------------------------------------------
    def attemptAtRasterizingVariableSizeCellsAndSavePIF(self, pFileName):

        # start progress bar:
        self.progressBar.setValue(0)
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        # we don't use a fixed size raster, so don't draw a grid on the image label:
        self.theRasterizedImageLabel.drawFixedSizeRaster(False)

        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: attemptAtRasterizingVariableSizeCellsAndSavePIF() to" + \
            str(pFileName) + " starting.", CDConstants.DebugExcessive )

        # open output file, and make sure that it's writable:
        lFile = QtCore.QFile(pFileName)
        if not lFile.open( QtCore.QFile.WriteOnly | QtCore.QFile.Text):
            CDConstants.printOut("___ - DEBUG ----- CDSceneRasterizer: attemptAtRasterizingVariableSizeCellsAndSavePIF() cannot write file" + \
                str(pFileName)+" : "+str(lFile.errorString())+" done.", CDConstants.DebugImportant )
            QtGui.QMessageBox.warning(self, self.tr("Recent Files"),
                    self.tr("Cannot write file %1:\n%2.").arg(pFileName).arg(lFile.errorString()))
            return False

        # open a QTextStream, i.e. an "interface for reading and writing text":
        lOutputStream = QtCore.QTextStream(lFile)
        # show the user that the application is busy (while writing to a file):
        # 2011 - Mitja: setOverrideCursor doesn't always restore to normal (on some platrforms?) so we don't change cursor for now:
        # QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)


        # This used to be set to render the width&height of the scene contents.
        # But we now set what we want as PIFF width & height separately in preferences, so we don't set it thus:
        #    lSceneWidth = self.theRasterizedImageLabel.width
        #    lSceneHeight = self.theRasterizedImageLabel.height
        # Instead we take PIFF width & height from the preferences'values as set to the graphics scene,
        #    which we rely on having been assigned to this object's local pointer copy:
        # lSceneWidth = self.theGraphicsScene.sceneRect().width()
        # lSceneHeight = self.theGraphicsScene.sceneRect().height()
        lSceneWidth = self.cdPreferences.pifSceneWidth
        lSceneHeight = self.cdPreferences.pifSceneHeight

        # ------------------------------------------------------------
        # now create a table of total cell type probabilities for each region i.
        #   as set in the self.regionsDict[lRegionsKeys[i]][4][j] dicts for each cell type j:
        lRegionsKeys = self.regionsDict.keys()
        lProbabilityTotalsDict = dict()

        for i in xrange(len(self.regionsDict)):
            for j in xrange( len(self.regionsDict[lRegionsKeys[i]][4]) ):
                # print "at i =", i, ", j =", j, \
                #       "self.regionsDict[keys[i]][4][j] =", \
                #        self.regionsDict[lRegionsKeys[i]][4][j]
                try:
                    soFar = lProbabilityTotalsDict[lRegionsKeys[i]]
                    lProbabilityTotalsDict[lRegionsKeys[i]] = soFar + self.regionsDict[lRegionsKeys[i]][4][j][2]
                except:
                    lProbabilityTotalsDict[lRegionsKeys[i]] = self.regionsDict[lRegionsKeys[i]][4][j][2]
                # print "lProbabilityTotalsDict[lRegionsKeys[i]] = ", \
                #     lProbabilityTotalsDict[lRegionsKeys[i]]
        # ------------------------------------------------------------

        # start progress bar:
        # self.progressBar.setRange(0, ( lSceneWidthInPixels * lSceneHeightInPixels ) )
        # for the progress bar we just increment once per processed item:
        lSceneItemsCount = len(self.theGraphicsScene.items())
        lSceneProcessedItem = 0
        self.progressBar.setRange(lSceneProcessedItem, lSceneItemsCount - 1)

        lSceneItems = self.theGraphicsScene.items(QtCore.Qt.AscendingOrder)

        lRegionsKeys = self.regionsDict.keys()

        lCellID = 0

        for lItem in lSceneItems:

            lColor = lItem.brush().color().rgba()
            lRegionKey = self.colorToKeyRegionDict[lColor]

            lItemsSceneRect = lItem.sceneBoundingRect()
            lItemsRect = lItem.boundingRect()
            lItemWidth = lItemsSceneRect.width()
            lItemHeight = lItemsSceneRect.height()
            lItemLocalLeft = lItemsRect.left()
            lItemLocalRight = lItemsRect.right()
            lItemLocalTop = lItemsRect.top()
            lItemLocalBottom = lItemsRect.bottom()

            # rasterize region items differently from cell items in the Cell Scene:
            if lItem.itsaRegionOrCell is CDConstants.ItsaRegionConst:

                # ---------- CDConstants.ItsaRegionConst ----------
                CDConstants.printOut( "rasterizing CDConstants.ItsaRegionConst: "+str(lItem), CDConstants.DebugAll )

                lItemsRasterSize = self.colorToCellSizeRegionDict[lColor]

                CDConstants.printOut( str("for i in range(int(lItemLocalLeft=%s), int(lItemLocalRight=%s), lItemsRasterSize=%s):" % \
                    (lItemLocalLeft, lItemLocalRight, lItemsRasterSize)), CDConstants.DebugAll )

                for x in range(int(lItemLocalLeft), int(lItemLocalRight), lItemsRasterSize):

                    CDConstants.printOut( str("    for y in range(int(lItemLocalTop=%s), int(lItemLocalBottom=%s), lItemsRasterSize=%s):" % \
                        (lItemLocalTop, lItemLocalBottom, lItemsRasterSize)), CDConstants.DebugAll )


                    for y in range(int(lItemLocalTop), int(lItemLocalBottom), lItemsRasterSize):

                        # translate the sampled point to object coordinates for sampling:
                        lPoint = QtCore.QPointF(x,y) ##### - lItem.scenePos()

                        if lItem.contains( lPoint ) :
                            CDConstants.printOut( str("INSIDE x=%s y=%s L=%s R=%s T=%s B=%s\n" % \
                                (x, y, lItemLocalLeft, lItemLocalRight, lItemLocalTop, lItemLocalBottom)), CDConstants.DebugAll )
                            lColor = lItem.brush().color().rgba()
                        else:
                            CDConstants.printOut( "                                  outside", CDConstants.DebugAll )
                            lColor = QtGui.QColor(QtCore.Qt.transparent).rgba()

                        if lColor in self.colorToNameRegionDict:
                            lPointInScene = lItem.sceneTransform().map(QPointF(x, y))
                            xmin = int(lPointInScene.x())
                            ymin = int((lSceneHeight-1) - lPointInScene.y())
                            xmax = xmin+lItemsRasterSize-1
                            ymax = ymin+lItemsRasterSize-1

                            # generate a random floating point number between 0.0 and the region's total probability:
                            lRnd = random.random()
                            lRnd = lRnd * lProbabilityTotalsDict[lRegionKey]
       
                            # loop through all cell type dicts for the current region until probability is matched:
                            lTheCellTypeName = ""
                            lRndCumulative = 0.0
                            for k in xrange(len(  self.regionsDict[lRegionKey][4]  )) :
                                try:
                                    lRndCumulative = lRndCumulative + self.regionsDict[lRegionKey][4][k][2]
                                except:
                                    lRndCumulative = self.regionsDict[lRegionKey][4][k][2]
                                # if the cell type's probability is matched by the random number, get the cell type name:
                                if ( (lTheCellTypeName == "") and (lRndCumulative > lRnd) ) :
                                    lTheCellTypeName = self.regionsDict[lRegionKey][4][k][1]
                            CDConstants.printOut( str("%s %s %s %s %s %s 0 0\n"%(lCellID, lTheCellTypeName, xmin, xmax, ymin, ymax)), CDConstants.DebugAll )
                            lOutputStream << "%s %s %s %s %s %s 0 0\n"%(lCellID, lTheCellTypeName, xmin, xmax, ymin, ymax)
                            lCellID +=1
                        else:
                            # if we caught a color not in the dictionary, we output nothing to the file!
                            pass
            else:

                # ---------- CDConstants.ItsaCellConst ----------
                CDConstants.printOut( "rasterizing CDConstants.ItsaCellConst:" + str(lItem), CDConstants.DebugAll )

                lItemsCellName = self.colorToNameRegionDict[lColor]

                for x in xrange(0, int(lSceneWidthInPixels), 1):
                    for y in xrange(0, int(lSceneHeightInPixels), 1):

                        lSceneWidthInPixels = self.theGraphicsScene.sceneRect().width()
                        lSceneHeightInPixels = self.theGraphicsScene.sceneRect().height()

            # progressBar status update:
            lSceneProcessedItem = lSceneProcessedItem + 1
            self.progressBar.setValue(lSceneProcessedItem)
            # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
            #   unless we also explicitly ask Qt to process at least some events.
            QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)


        # end progress bar:
        self.progressBar.setValue( lSceneItemsCount - 1 )
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        # cleanly close access to the file:
        lFile.close()

        CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: attemptAtRasterizingVariableSizeCellsAndSavePIF() PIFF file saving to" + \
            str(pFileName) + " done." , CDConstants.DebugExcessive )

        # 2011 - Mitja: this doesn't always restore to normal (on different platrforms?) so we don't change cursor for now:
        # QtGui.QApplication.restoreOverrideCursor()


    # end of def attemptAtRasterizingVariableSizeCellsAndSavePIF(self)
    # ---------------------------------------------------------





    # ---------------------------------------------------------
    # set up paths to access CC3D as a subprocess:
    # ---------------------------------------------------------
    def setupPathsToCC3D(self):

        # ---------------------------------------
        # 2011 - Mitja: add calling CC3D as subprocess:
        
        # make sure that preferences obtained from CC3D settings file are up to date:
        self.cdPreferences.readCC3DPreferencesFromDisk()

        self.cc3dPath = str(self.cdPreferences.cc3dCommandPathCC3D)
        self.cc3dPathAndStartupFileName = str(self.cdPreferences.cc3dCommandPathAndStartCC3D)
        
        self.cc3dOutputLocationPath = str(self.cdPreferences.outputLocationPathCC3D)


        # TODO TODO: is self.pluginObj necessary for anything?
        self.pluginObj=None

        self.cc3dProcess=None
   
    # end of def setupPathsToCC3D(self)
    # ---------------------------------------------------------


    # ---------------------------------------------------------
    # start CC3D as a subprocess:
    # ---------------------------------------------------------
    def startCC3D(self,_simulationName=""):
        from subprocess import Popen   

        CDConstants.printOut( "self.cc3dPath=" + str(self.cc3dPath), CDConstants.DebugAll )
        CDConstants.printOut( "self.cc3dPathAndStartupFileName=" + str(self.cc3dPathAndStartupFileName), CDConstants.DebugAll )
        CDConstants.printOut( "self.cc3dOutputLocationPath=" + str(self.cc3dOutputLocationPath), CDConstants.DebugAll )
        CDConstants.printOut( "self.cdPreferences.cellDrawDirectoryPath=" + str(self.cdPreferences.cellDrawDirectoryPath), CDConstants.DebugAll )

        # popenArgs=[self.cc3dPathAndStartupFileName,"--port=%s"%self.port]
        popenArgs=[self.cc3dPathAndStartupFileName]

        if _simulationName!="":
            popenArgs.append("-i")
            popenArgs.append(_simulationName)

        # popenArgs.append("-i")
        # popenArgs.append("D:\\Program Files\\COMPUCELL3D_3.5.1_install2\\examples_PythonTutorial\\infoPrinterDemo\\infoPrinterDemo.cc3d" )
       
        CDConstants.printOut( "popenArgs=" + str(popenArgs), CDConstants.DebugAll )
        self.cc3dProcess = Popen(popenArgs)           
       
        # self.cc3dProcess = Popen([self.cc3dPathAndStartupFileName,"--port=%s"%self.port])
       
        # ,"--tweditPID=%s"%self.editorWindow.getProcessId()

    # end of def startCC3D(self,_simulationName="")
    # ---------------------------------------------------------



    # ---------------------------------------------------------
    # convert the graphics items (cell regions) to a file that can be used by CC3D
    #    to transform from regions to cells using the Potts model,
    #    then it's read back here and saved as PIFF
    # ---------------------------------------------------------
    def computePottsModelAndSavePIF(self):

        # set up paths to access CC3D as a subprocess:
        self.setupPathsToCC3D()

        # start progress bar:
        self.progressBar.setValue(0)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)


        # we don't use a fixed size raster, so don't draw a grid on the image label:
        self.theRasterizedImageLabel.drawFixedSizeRaster(False)
        self.theRasterizedImageLabel.update()

        # ------------------------------------------------------------
        # (1) - computePottsModelAndSavePIF - (1)
        #
        #       setup overall parameters for the Cell Scene:
        #
        # This used to be set to render the width&height of the scene contents.
        # But we now set what we want as PIFF width & height separately in preferences, so we don't set it thus:
        #    lSceneWidthInPixels = self.theRasterizedImageLabel.width
        #    lSceneHeightInPixels = self.theRasterizedImageLabel.height
        # Instead we take PIFF width & height from the preferences'values as set to the graphics scene,
        #    which we rely on having been assigned to this object's local pointer copy:
        # lSceneWidthInPixels = self.theGraphicsScene.sceneRect().width()
        # lSceneHeightInPixels = self.theGraphicsScene.sceneRect().height()

        lSceneWidthInPixels = self.cdPreferences.pifSceneWidth
        lSceneHeightInPixels = self.cdPreferences.pifSceneHeight

        # 2011 - Mitja: create an empty array, into which to write pixel values,
        #    one for each pixel (not just one for each cell!) :

        self.scenePixelsArray = numpy.zeros( \
            (lSceneHeightInPixels, lSceneWidthInPixels), \
            dtype=numpy.int32 )


        # ------------------------------------------------------------
        # (2) - computePottsModelAndSavePIF - (2)
        #
        #       create a table of total cell type probabilities for each region i
        #         as set in the self.regionsDict[lRegionsKeys[i]][4][j] dicts
        #         for each cell type j:
        #
        lRegionsKeys = self.regionsDict.keys()
        lProbabilityTotalsDict = dict()
        CDConstants.printOut( "lRegionsKeys = self.regionsDict.keys() =" + str(lRegionsKeys), CDConstants.DebugAll )
        CDConstants.printOut( "self.colorToKeyRegionDict =" + str(self.colorToKeyRegionDict), CDConstants.DebugAll )
       
        for i in xrange(len(self.regionsDict)):
            for j in xrange( len(self.regionsDict[lRegionsKeys[i]][4]) ):
                # print "at i =", i, ", j =", j, \
                #       "self.regionsDict[keys[i]][4][j] =", \
                #        self.regionsDict[lRegionsKeys[i]][4][j]

                try:
                    soFar = lProbabilityTotalsDict[lRegionsKeys[i]]
                    lProbabilityTotalsDict[lRegionsKeys[i]] = soFar + \
                        self.regionsDict[lRegionsKeys[i]][4][j][2]
                except:
                    # the first time through the loop we land here:
                    lProbabilityTotalsDict[lRegionsKeys[i]] = \
                        self.regionsDict[lRegionsKeys[i]][4][j][2]

                CDConstants.printOut( "lProbabilityTotalsDict[lRegionsKeys[i="+str(i)+"]="+ \
                    str(lRegionsKeys[i])+"] = "+str(lProbabilityTotalsDict[lRegionsKeys[i]]), CDConstants.DebugExcessive )
        # ------------------------------------------------------------
        # now lProbabilityTotalsDict[] contains for each region a total probability value
        # ------------------------------------------------------------



        # ------------------------------------------------------------
        # (3) - computePottsModelAndSavePIF - (3)
        #
        #       start a loop over each region (item) of cells in the Cell Scene:
        #
        lSceneItems = self.theGraphicsScene.items(QtCore.Qt.AscendingOrder)

        # set progress bar:
        self.progressBar.setRange(0, len(lSceneItems) )
        self.progressBar.setValue(0)

        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        CDConstants.printOut( "", CDConstants.DebugAll )
        CDConstants.printOut( "computePottsModelAndSavePIF - running conversion to numpy array....", CDConstants.DebugAll )
        lCellID = 0
        lItemCounter = 0

        # ------------------------------------------------------------
        for lItem in lSceneItems:

            # ------------------------------------------------------------
            # update the progressBar to provide visual feedback to the user:
            self.progressBar.setValue(lItemCounter + 1)
            # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
            #   unless we also explicitly ask Qt to process at least some events.
            QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)


            # ------------------------------------------------------------
            # (4) - computePottsModelAndSavePIF - (4)
            #
            # from the lItem region (a QGraphicsItem) obtain its sceneTransform,
            #   and relevant information about its region key:

            lItemsColor = lItem.brush().color().rgba()
            lItemsSceneTransform = lItem.sceneTransform()

            # for Potts-model generation of PIFF regions we don't use a single size per region:
            # lItemsRasterSize = self.colorToCellSizeRegionDict[lItemsColor]

            #  colorToKeyRegionDict is used to map color RGBA values
            #     to region keys (integers starting from 1):
            lRegionKey = self.colorToKeyRegionDict[lItemsColor]

            # map from scene correctly, i.e. using inverse scene transform
            #   for translation/rotation/scaling of the object:
            #   (NOTE: in PyQt, inverted() returns a tuple, not a single value)
            lItemInverseTransform,lIsNotSingular = lItemsSceneTransform.inverted()
            if lIsNotSingular is False:
                QtGui.QMessageBox.warning(self, "CellDraw", \
                    self.tr("Can't use scene item %1: singular QTransform.").arg(lItem))
                return False
            else:
                CDConstants.printOut( "lItemInverseTransform = \n" + \
                " - - - - - " + str(lItemInverseTransform.m11()) +" "+ str(lItemInverseTransform.m21()) +" "+ str(lItemInverseTransform.m31()) + "\n" + \
                " - - - - - " + str(lItemInverseTransform.m12()) +" "+ str(lItemInverseTransform.m22()) +" "+ str(lItemInverseTransform.m32()) + "\n" + \
                " - - - - - " + str(lItemInverseTransform.m13()) +" "+ str(lItemInverseTransform.m23()) +" "+ str(lItemInverseTransform.m33()) + "\n", CDConstants.DebugAll )

                lBoundingRect = lItem.boundingRegion(lItem.sceneTransform()).boundingRect()
                CDConstants.printOut( "boundingRegion-sceneTransform-width(), boundingRegion-sceneTransform-height() = " + \
                    str(lBoundingRect.width()) + " " +str(lBoundingRect.height()), CDConstants.DebugAll )
#                 lBoundingRect2 = lItem.polygon().boundingRegion(lItem.sceneTransform()).boundingRect()
#                 print "boundingRegion-polygon-sceneTransform-width(), boundingRegion-polygon-sceneTransform-height() = ", lBoundingRect2.width(), lBoundingRect2.height()


                CDConstants.printOut( "lItem.boundingRect().width() = " + str(lItem.boundingRect().width()) , CDConstants.DebugAll )
                CDConstants.printOut( "lItem.boundingRect().height() = " + str(lItem.boundingRect().height()) , CDConstants.DebugAll )
                CDConstants.printOut( "lItem.polygon().boundingRect().width() = " + str(lItem.polygon().boundingRect().width()) , CDConstants.DebugAll )
                CDConstants.printOut( "lItem.polygon().boundingRect().height() = " + str(lItem.polygon().boundingRect().height()) , CDConstants.DebugAll )


            # ------------------------------------------------------------
            # (5) - computePottsModelAndSavePIF - (5)
            #
            # obtain volume and percentage properties for all cell types in this region:

            lCellsVolumes = dict()
            lCellsPercentage = dict()

            lNumberOfCellTypes = len(self.regionsDict[lRegionKey][4])

            for j in xrange(lNumberOfCellTypes):
                lVolume = float(self.regionsDict[lRegionKey][4][j][3])
                lCellsVolumes[j] = lVolume
                CDConstants.printOut( "lCellsVolumes[j="+str(j)+" ] = "+str(lCellsVolumes[j]) , CDConstants.DebugAll )

                lPercentage = float(self.regionsDict[lRegionKey][4][j][2]) /   \
                    float(lProbabilityTotalsDict[lRegionKey])
                lCellsPercentage[j] = float(lPercentage)
                CDConstants.printOut( "lCellsPercentage[j="+str(j)+" ] = "+str(lCellsPercentage[j]) , CDConstants.DebugAll )


            # ------------------------------------------------------------
            # (6) - computePottsModelAndSavePIF - (6)
            #
            # compute the volume (i.e. area) of this region, by drawing it
            #    and counting the pixels!
            #
            lRegionVolumeInPixels = 0.0
            lTmpPixmap = QtGui.QPixmap(lSceneWidthInPixels, lSceneHeightInPixels)
            lTmpPixmap.fill(QtCore.Qt.transparent)
            lTmpPainter = QtGui.QPainter(lTmpPixmap)
            lTmpPainter.setWorldTransform(lItemsSceneTransform)
            lTmpPen = QtGui.QPen(QtCore.Qt.transparent, 1)
            lTmpPen.setCosmetic(False)
            lTmpPainter.setPen(lTmpPen)
            lTmpBrush = QtGui.QBrush(QtGui.QColor(lItemsColor))
            lTmpPainter.setBrush(lTmpBrush)
            #  self.setFillRule(QtCore.Qt.WindingFill) from Qt documentation:
            # Specifies that the region is filled using the non zero winding rule.
            # With this rule, we determine whether a point is inside the shape by
            # using the following method. Draw a horizontal line from the point to a
            # location outside the shape. Determine whether the direction of the line
            # at each intersection point is up or down. The winding number is
            # determined by summing the direction of each intersection. If the number
            # is non zero, the point is inside the shape. This fill mode can also in
            # most cases be considered as the intersection of closed shapes.
            lTmpPainter.drawPolygon(lItem.polygon(), QtCore.Qt.WindingFill)
            lTmpPainter.end()
            # provide visual feedback to user:
            self.theRasterizedImageLabel.drawPixmapAtPoint(lTmpPixmap)
            self.theRasterizedImageLabel.update()
            # copy the QPixmap into a QImage, since QPixmap's pixels *can't* be accessed:
            lTmpImage = QtGui.QImage(lTmpPixmap.toImage())
            for i in xrange(0, int(lSceneWidthInPixels)):
                for j in xrange(0, int(lSceneHeightInPixels)):
                    # grab the QRgb value at position (x=i, y=j) in the Qimage:
                    lRGBAColorAtClickedPixel = lTmpImage.pixel(i, j)
                    if (lRGBAColorAtClickedPixel == lItemsColor):
                        lRegionVolumeInPixels = lRegionVolumeInPixels + 1.0
            # finally got the volume (i.e. how many pixels) of the entire item/region:
            CDConstants.printOut( "lRegionVolumeInPixels = " +str(lRegionVolumeInPixels), CDConstants.DebugAll )

# compute these correctly:
#             lCellVolume = float(lItemsRasterSize * lItemsRasterSize)
#             lBoundingRect = lItem.boundingRegion(lItem.sceneTransform()).boundingRect()
#             lRegionVolumeInPixels = float(lBoundingRect.width() * lBoundingRect.height())


            # ------------------------------------------------------------
            # (7) - computePottsModelAndSavePIF - (7)
            #
            # compute the number of cells (i.e. required starting points)
            #    for each type of cell in this region:
            #    by A. computing the [ %a * Va + %b * Vb + ... ] total for all cells
            #    and B. dividing the region volume by that total:

            lCellsFractionsVolumesTotal = 0.0
            lFractVolumePerCellType = 0.0
            for j in xrange(lNumberOfCellTypes):
                lFractVolumePerCellType = lCellsVolumes[j] * lCellsPercentage[j]
                lCellsFractionsVolumesTotal = \
                    lCellsFractionsVolumesTotal + lFractVolumePerCellType
                CDConstants.printOut( "lCellsFractionsVolumesTotal = "+str(lCellsFractionsVolumesTotal), CDConstants.DebugAll )
            # add 10% to cell number, since some cells may die out ...
            lRequiredCellPoints =  1.1 * \
                ( lRegionVolumeInPixels / lCellsFractionsVolumesTotal )

            # ------------------------------------------------------------
            # (8) - computePottsModelAndSavePIF - (8)
            #
            # now that we have all initially required data,
            #   we draw the item into a pixmap of the same size as the Cell Scene,
            #   so that its boundary can be dumped into the numpy array:
            #
            lTmpPixmap = QtGui.QPixmap(lSceneWidthInPixels, lSceneHeightInPixels)
            lTmpPixmap.fill(QtCore.Qt.transparent)
            lTmpPainter = QtGui.QPainter(lTmpPixmap)
            lTmpPainter.setWorldTransform(lItemsSceneTransform)
            lTmpPen = QtGui.QPen(QtCore.Qt.black)
            lTmpPen.setWidth(2)
            lTmpPen.setCosmetic(True)
            lTmpPainter.setPen(lTmpPen)
            lTmpBrush = QtGui.QBrush(QtGui.QColor(QtCore.Qt.white))
            lTmpPainter.setBrush(lTmpBrush)
            #  self.setFillRule(QtCore.Qt.WindingFill) from Qt documentation:
            # Specifies that the region is filled using the non zero winding rule.
            # With this rule, we determine whether a point is inside the shape by
            # using the following method. Draw a horizontal line from the point to a
            # location outside the shape. Determine whether the direction of the line
            # at each intersection point is up or down. The winding number is
            # determined by summing the direction of each intersection. If the number
            # is non zero, the point is inside the shape. This fill mode can also in
            # most cases be considered as the intersection of closed shapes.
            lTmpPainter.drawPolygon(lItem.polygon(), QtCore.Qt.WindingFill)
            lTmpPainter.end()
            # provide visual feedback to user:
            self.theRasterizedImageLabel.drawPixmapAtPoint(lTmpPixmap)
            self.theRasterizedImageLabel.update()

            # ------------------------------------------------------------
            # (9) - computePottsModelAndSavePIF - (9)
            #
            # plot all cells' initial points on the top of the boundary-drawn area:
            #
            # (9a) - compute all the cell initial points' locations into a list:
            lDone = False
            lCellPoints = []
            lTmpImage = QtGui.QImage(lTmpPixmap.toImage())
            lWallColor = QtGui.QColor(QtCore.Qt.black).rgba()
            while not lDone:
                lRandomPointOK = False
                while not lRandomPointOK:
                    lRndX = random.random()
                    lRndX = lRndX * float(lSceneWidthInPixels - 1)
                    i = int(lRndX)
                    lRndY = random.random()
                    lRndY = lRndY * float(lSceneHeightInPixels - 1)
                    j = int(lRndY)
                    lPoint = lItemInverseTransform.map( QtCore.QPointF(lRndX,lRndY) )
                    # make sure that the point coordinates don't end up on the wall boundary:
                    if lItem.contains( lPoint ) and (lTmpImage.pixel(i, j) != lWallColor):
                        lCellPoints.append( (i,j) )
                        # print "lCellPoints = ", lCellPoints
                        lRandomPointOK = True
                if len(lCellPoints) >= lRequiredCellPoints:
                    lDone = True

            # (9b) - plot all the cell initial points into the temporary pixmap:
            lTmpColor = QtGui.QColor()
            lTmpColor.setRgba(lItemsColor)
            lTmpPen = QtGui.QPen()
            lTmpPen.setColor(lTmpColor)
            lTmpPen.setWidth(1)
            lTmpPen.setCosmetic(True)
            lTmpPainter = QtGui.QPainter(lTmpPixmap)
            lTmpPainter.setPen(lTmpPen)
            for (i,j) in lCellPoints :
                lTmpPainter.drawPoint(i,j)
            lTmpPainter.end()
            # provide visual feedback to user:
            self.theRasterizedImageLabel.drawPixmapAtPoint(lTmpPixmap)
            self.theRasterizedImageLabel.update()

            # ------------------------------------------------------------
            # (10) - computePottsModelAndSavePIF - (10)
            #
            # copy the item's data from the QImage into the numpy array, where:
            #   - the item/region border (black) becomes -1
            #   - the empty medium (transparent) does not get written into the array
            #   - the place where Potts will grow cells (white) becomes 0 (and then empty):
            #   - the items/celltypes become ((1000 * item_region_key) + celltype_key)
            #
            # copy the QPixmap into a QImage, since QPixmap's pixels *can't* be accessed:
            lTmpImage = QtGui.QImage(lTmpPixmap.toImage())

            lNumOfPointsForThisRegion = 0
            for i in xrange(0, int(lSceneWidthInPixels)):
                for j in xrange(0, int(lSceneHeightInPixels)):
               
                    # grab the QRgb value at position (x=i, y=j) in the Qimage:
                    lRGBAColorAtClickedPixel = lTmpImage.pixel(i, j)


                    # store the appropriate value in a numpy array:
                    #  (strangely enough, for numpy arrays the 1st parameter is rows (y)
                    #       and the 2nd parameter is columns (x)   ) :
                    if (lRGBAColorAtClickedPixel == QtGui.QColor(QtCore.Qt.black).rgba()):
                        # the item/region border (black) becomes -1 :
                        self.scenePixelsArray[j, i] = -1
                    elif (lRGBAColorAtClickedPixel == QtGui.QColor(QtCore.Qt.transparent).rgba()):
                        # empty medium (transparent) does not get written into the array:
                        pass
                    elif (lRGBAColorAtClickedPixel == QtGui.QColor(QtCore.Qt.white).rgba()):
                        # the place where Potts will grow cells (white) becomes 0:
                        self.scenePixelsArray[j, i] = 0
                    else:
                        # the items/celltypes become ((1000 * item_region_key) + celltype_key) :
                        # ------------------------------------------------------------
                        # generate a random floating point number: [0.0 ... region's tot probability]:
                        lRegionKey = self.colorToKeyRegionDict[lItemsColor]
                        lRnd = random.random()
                        lRnd = lRnd * lProbabilityTotalsDict[lRegionKey]
       
                        # loop through all cell types for current region until probability is matched:
                        lTheCellTypeName = ""
                        lTheCellTypeKey = -1
                        lRndCumulative = 0.0
                        lNumberOfCellTypes = len(self.regionsDict[lRegionKey][4])
                        for k in xrange(lNumberOfCellTypes) :
                            try:
                                lRndCumulative = lRndCumulative + self.regionsDict[lRegionKey][4][k][2]
                            except:
                                lRndCumulative = self.regionsDict[lRegionKey][4][k][2]
                            # if cell type's probability is matched by random number, get its name,key:
                            if ( (lTheCellTypeName == "") and (lRndCumulative > lRnd) ) :
                                lTheCellTypeName = self.regionsDict[lRegionKey][4][k][1]
                                lTheCellTypeKey = k
                        # ------------------------------------------------------------
                        self.scenePixelsArray[j, i] = ( (1000 * lRegionKey) + lTheCellTypeKey )
                        lCellID = lCellID + 1
                        lNumOfPointsForThisRegion = lNumOfPointsForThisRegion + 1

                    # print "lPixelInLabel ("+str(i)+","+str(j)+") = "+lPixelInLabel

            lItemCounter = lItemCounter + 1
            CDConstants.printOut( "Generated numpy array with "+str(lNumOfPointsForThisRegion)+ \
                " points for item_region "+str(lItemCounter)+" : "+str(lRegionKey) , CDConstants.DebugExcessive )

            #self.infoLabel.setText("123456789012345678901234567890123456789012345678901234567890")
            self.infoLabel.setText( self.tr("Generated cell data array for item %1 (cell region %2)").arg(lItemCounter).arg(lRegionKey) )

            time.sleep(2.0)

        # ------------------------------------------------------------
        # end of  for lItem in lSceneItems
        # ------------------------------------------------------------
        CDConstants.printOut( "Generated numpy array with "+str(lCellID)+" total points for the Cell Scene.", CDConstants.DebugExcessive )



        

        # ------------------------------------------------------------
        # (11) - computePottsModelAndSavePIF - (11)
        #
        # copy the PIFF data from the numpy array into a file, where:
        #   - the item/region border (black) -1 becomes "Wall" cells
        #   - the empty medium (transparent) 0 does not get written out
        #   - the items/celltypes ((1000 * item_region_key) + celltype_key) become 1x1 pixels
        #
        # ------------------------------------------------------------

        # first clean, then create the directory where we place intermediate/temporary files to/from CC3D :
        lHelperOutputDirectoryCC3D = os.path.join(self.cc3dOutputLocationPath, "cellDrawHelpFiles")
        if  os.path.isdir(lHelperOutputDirectoryCC3D):
            print "=====>=====> \"cellDrawHelpFiles\" directory exists... Removing " + str(lHelperOutputDirectoryCC3D) + " and creating new directory."
            shutil.rmtree(lHelperOutputDirectoryCC3D)
        os.mkdir(lHelperOutputDirectoryCC3D)
        lHelperSimulationDirectoryCC3D = os.path.join(lHelperOutputDirectoryCC3D, "Simulation")
        os.mkdir(lHelperSimulationDirectoryCC3D)

        # grab the default helper files from our own CellDraw sourcecode directory,
        #   and copy them over to the temporary work directory:
        shutil.copy(  os.path.join(self.cdPreferences.cellDrawDirectoryPath,"cc3Dhelpfiles/helpfile_CellDraw.cc3d"), \
            os.path.join(lHelperOutputDirectoryCC3D,"helpfile_CellDraw.cc3d")  )
        shutil.copy(  os.path.join(self.cdPreferences.cellDrawDirectoryPath,"cc3Dhelpfiles/Simulation/helpfile_CellDraw.py"), \
            os.path.join(lHelperSimulationDirectoryCC3D,"helpfile_CellDraw.py")  )
        shutil.copy(  os.path.join(self.cdPreferences.cellDrawDirectoryPath,"cc3Dhelpfiles/Simulation/helpfile_steppables_CellDraw.py"), \
            os.path.join(lHelperSimulationDirectoryCC3D,"helpfile_steppables_CellDraw.py")  )

        lHelperPIFFileName=os.path.join(lHelperSimulationDirectoryCC3D,"helpfile.piff")
        CDConstants.printOut( "=====>=====> lHelperPIFFileName = " + str(lHelperPIFFileName), CDConstants.DebugExcessive )
        lHelperPIFFile = QtCore.QFile(lHelperPIFFileName)
        lOnlyThePathHelperPIFFileName,lOnlyTheFileHelperPIFFileName = os.path.split(str(lHelperPIFFileName))
        if not lHelperPIFFile.open( QtCore.QFile.WriteOnly | QtCore.QFile.Text):
            QtGui.QMessageBox.warning(self, "CellDraw", \
                self.tr("Cannot write file %1 .\nError: [%2] .\n[in computePottsModelAndSavePIF() - (11a)]").arg(lOnlyTheFileHelperPIFFileName).arg(lHelperPIFFile.errorString()))
            self.hide()
            return False
        else:
            #self.infoLabel.setText("123456789012345678901234567890123456789012345678901234567890")
            self.infoLabel.setText( self.tr("Temporary Potts model saved to PIFF file:  %1").arg(lOnlyTheFileHelperPIFFileName) )
        # open a QTextStream, i.e. an "interface for reading and writing text":
        lHelperPIFFileOutputStream = QtCore.QTextStream(lHelperPIFFile)


        lCellID = 0
        for i in xrange(0, int(lSceneWidthInPixels)):
            for j in xrange(0, int(lSceneHeightInPixels)):
                # grab the value at position (x=i, y=j) in the numpy array:
                #  (strangely enough, for numpy arrays the 1st parameter is rows (y)
                #       and the 2nd parameter is columns (x)   ) :
                lPixel = self.scenePixelsArray[j, i]

                if (lPixel == 0):
                    # empty medium (transparent) does not get written into the array:
                    pass
                elif (lPixel == -1):
                    # the item/region border (black) becomes -1 :
                    lPixelCellTypeName = "Wall"
                    xmin = i
                    ymin = j
                    # print "%s %s %s %s %s %s 0 0\n"%(lCellID, lPixelCellTypeName, xmin, xmin, ymin, ymin)
                    lHelperPIFFileOutputStream << "%s %s %s %s %s %s 0 0\n"%(lCellID, lPixelCellTypeName, xmin, xmin, ymin, ymin)
                    lCellID +=1
                else:
                    # the items/celltypes become ((1000 * item_region_key) + celltype_key) :
                    lPixelRegionKey = int (lPixel / 1000)
                    lPixelCellTypeKey = int (lPixel  - (lPixelRegionKey*1000))
                    lPixelCellTypeName = self.regionsDict[lPixelRegionKey][4][lPixelCellTypeKey][1]
                    xmin = i
                    ymin = j
                    # print "%s %s %s %s %s %s 0 0\n"%(lCellID, lPixelCellTypeName, xmin, xmin, ymin, ymin)
                    lHelperPIFFileOutputStream << "%s %s %s %s %s %s 0 0\n"%(lCellID, lPixelCellTypeName, xmin, xmin, ymin, ymin)
                    lCellID +=1

        lHelperPIFFile.close()




        lHelperXMLFileName=os.path.join(lHelperSimulationDirectoryCC3D,"helpfile_cellDraw.xml")
        CDConstants.printOut( "=====>=====> lHelperXMLFileName = " + str(lHelperXMLFileName) , CDConstants.DebugExcessive )
        lHelperXMLFile = QtCore.QFile(lHelperXMLFileName)
        lOnlyThePathHelperXMLFileName,lOnlyTheFileHelperXMLFileName = os.path.split(str(lHelperXMLFileName))
        if not lHelperXMLFile.open( QtCore.QFile.WriteOnly | QtCore.QFile.Text):
            QtGui.QMessageBox.warning(self, "CellDraw", \
                self.tr("Cannot write file %1 .\nError: [%2] .\n[in computePottsModelAndSavePIF() - (11b)]").arg(lOnlyTheFileHelperXMLFileName).arg(lHelperXMLFile.errorString()))
            self.hide()
            return False
        else:
            #self.infoLabel.setText("123456789012345678901234567890123456789012345678901234567890")
            self.infoLabel.setText( self.tr("Saving temporary Potts model to XML file: %1").arg(lOnlyTheFileHelperXMLFileName) )

        # open a QTextStream, i.e. an "interface for reading and writing text":
        lHelperXMLFileOutputStream = QtCore.QTextStream(lHelperXMLFile)

        lHelperXMLFileOutputStream << "<CompuCell3D>\n"
        lHelperXMLFileOutputStream << "\n"
        lHelperXMLFileOutputStream << " <Potts>\n"
        lHelperXMLFileOutputStream << "   <Dimensions x=\"%s\" y=\"%s\" z=\"1\"/>\n" % (str(lSceneWidthInPixels), str(lSceneHeightInPixels))
        lHelperXMLFileOutputStream << "   <Anneal>0</Anneal>\n"
        lHelperXMLFileOutputStream << "   <Steps>220</Steps>\n"
        lHelperXMLFileOutputStream << "   <Temperature>10</Temperature>\n"
        lHelperXMLFileOutputStream << "   <Flip2DimRatio>1</Flip2DimRatio>\n"
        lHelperXMLFileOutputStream << "   <DebugOutputFrequency>0</DebugOutputFrequency>\n"
        lHelperXMLFileOutputStream << " </Potts>\n"
#         lHelperXMLFileOutputStream << "\n"
#         lHelperXMLFileOutputStream << " <Plugin Name=\"PlayerSettings\">\n"
#         lHelperXMLFileOutputStream << "    <Rotate3D XRot=\"27\" YRot=\"-11\"/>\n"
#         lHelperXMLFileOutputStream << " </Plugin>\n"
        lHelperXMLFileOutputStream << "\n"
        lHelperXMLFileOutputStream << " <Plugin Name=\"CellType\">\n"
        lHelperXMLFileOutputStream << "    <CellType TypeName=\"Medium\" TypeId=\"0\"/>\n"
        lHelperXMLFileOutputStream << "    <CellType TypeName=\"Wall\" TypeId=\"1\" Freeze=\"\"/>\n"
       
        # for .... find out type names and put out some sequential ID numbers...

        lRegionsKeys = self.regionsDict.keys()
        lUsedCellTypeVolumesDict = dict()
        lUsedCellIDToNameDict = dict()
        lUsedCellIDToColorDict = dict()        
        lTypeID = 2

        for i in xrange(len(self.regionsDict)):
            # test that the i-th region is in use
            #   i.e. the entry 3 in a regionsDict item is a counter of how many Cell Scene regions use this color:
            if (self.regionsDict[lRegionsKeys[i]][3] > 0) :
                # print "self.regionsDict[lRegionsKeys[i=",i,"]=",lRegionsKeys[i],"] = ", self.regionsDict[lRegionsKeys[i]]
                for j in xrange( len(self.regionsDict[lRegionsKeys[i]][4]) ):
                    # get each used cell type's name and target volume size:
                    lCellTypeName = self.regionsDict[lRegionsKeys[i]][4][j][1]
                    lCellTypeVolume = self.regionsDict[lRegionsKeys[i]][4][j][3]
                    lCellTypeColor = QtGui.QColor(self.regionsDict[lRegionsKeys[i]][4][j][0])

                    # save four data for each cell type: its region ID, its type name, its target volume, and an incremental ID starting from 2:
                    lUsedCellTypeVolumesDict[((1+i)*1000)+(j+1)] = \
                        (lRegionsKeys[i],lCellTypeName,lCellTypeVolume,lTypeID)

                    # also save a separate dict to retrieve each cell type name from its ID:
                    lUsedCellIDToNameDict[lTypeID] = lCellTypeName
                    lUsedCellIDToColorDict[lTypeID] = lCellTypeColor
                    lTypeID = lTypeID+1

        CDConstants.printOut( "lUsedCellTypeVolumesDict = "+ str(lUsedCellTypeVolumesDict), CDConstants.DebugExcessive )
        CDConstants.printOut( "lUsedCellIDToNameDict ="+ str(lUsedCellIDToNameDict), CDConstants.DebugExcessive )

        # write to the output stream one entry per cell type, and an incremental type ID:
        lCellTypeKeys = lUsedCellTypeVolumesDict.keys()
        for i in xrange(len(lUsedCellTypeVolumesDict)):
            lRegionKey,lCellTypeName,lCellTypeVolume,lTypeID = lUsedCellTypeVolumesDict[lCellTypeKeys[i]]
            lHelperXMLFileOutputStream << "    <CellType TypeName=\"%s\" TypeId=\"%s\"/>\n" % (lCellTypeName,str(lTypeID))
            #lHelperXMLFileOutputStream << "    <CellType TypeName="NonCondensing" TypeId="3"/>\n"
        lHelperXMLFileOutputStream << " </Plugin>\n"

#  for the LambdaVolume and Energy parameters, this is what seems to work, testing with Gilberto on 2011.05.10:
#
#          <Plugin Name="VolumeFlex">
#             <VolumeEnergyParameters CellType="g1" TargetVolume="100.0" LambdaVolume="10"/>
#             <VolumeEnergyParameters CellType="g2" TargetVolume="200.0" LambdaVolume="10"/>
#             <VolumeEnergyParameters CellType="b1" TargetVolume="50.0" LambdaVolume="30"/>
#             <VolumeEnergyParameters CellType="b2" TargetVolume="10.0" LambdaVolume="30"/>
#             <VolumeEnergyParameters CellType="r1" TargetVolume="20.0" LambdaVolume="20"/>
#             <VolumeEnergyParameters CellType="r2" TargetVolume="40.0" LambdaVolume="15"/>
#          </Plugin>
#        
#          <Plugin Name="Contact">
#            <Energy Type1="Wall" Type2="Wall">0</Energy>
#            <Energy Type1="Wall" Type2="Medium">0</Energy>
#            <Energy Type1="Medium" Type2="Medium">0</Energy>
#            <Energy Type1="Wall" Type2="g1">50</Energy>
#            <Energy Type1="Medium" Type2="g1">16</Energy>
#            <Energy Type1="Wall" Type2="g2">50</Energy>
#            <Energy Type1="Medium" Type2="g2">16</Energy>
#            <Energy Type1="Wall" Type2="b1">50</Energy>
#            <Energy Type1="Medium" Type2="b1">16</Energy>
#            <Energy Type1="Wall" Type2="b2">50</Energy>
#            <Energy Type1="Medium" Type2="b2">16</Energy>
#            <Energy Type1="Wall" Type2="r1">50</Energy>
#            <Energy Type1="Medium" Type2="r1">16</Energy>
#            <Energy Type1="Wall" Type2="r2">50</Energy>
#            <Energy Type1="Medium" Type2="r2">16</Energy>
#            <Energy Type1="g1" Type2="g1">16</Energy>
#            <Energy Type1="g1" Type2="g2">16</Energy>
#            <Energy Type1="g1" Type2="b1">16</Energy>
#            <Energy Type1="g1" Type2="b2">16</Energy>
#            <Energy Type1="g1" Type2="r1">16</Energy>
#            <Energy Type1="g1" Type2="r2">16</Energy>
#            <Energy Type1="g2" Type2="g2">16</Energy>
#            <Energy Type1="g2" Type2="b1">16</Energy>
#            <Energy Type1="g2" Type2="b2">16</Energy>
#            <Energy Type1="g2" Type2="r1">16</Energy>
#            <Energy Type1="g2" Type2="r2">16</Energy>
#            <Energy Type1="b1" Type2="b1">16</Energy>
#            <Energy Type1="b1" Type2="b2">16</Energy>
#            <Energy Type1="b1" Type2="r1">16</Energy>
#            <Energy Type1="b1" Type2="r2">16</Energy>
#            <Energy Type1="b2" Type2="b2">16</Energy>
#            <Energy Type1="b2" Type2="r1">16</Energy>
#            <Energy Type1="b2" Type2="r2">16</Energy>
#            <Energy Type1="r1" Type2="r1">16</Energy>
#            <Energy Type1="r1" Type2="r2">16</Energy>
#            <Energy Type1="r2" Type2="r2">16</Energy>
#            <Depth>2</Depth>
#          </Plugin>

        # write to the output stream one entry per cell type, and its target volume:
        lHelperXMLFileOutputStream << "\n"
        lHelperXMLFileOutputStream << " <Plugin Name=\"VolumeFlex\">\n"
        for i in xrange(len(lUsedCellTypeVolumesDict)):
            lRegionKey,lCellTypeName,lCellTypeVolume,lTypeID = lUsedCellTypeVolumesDict[lCellTypeKeys[i]]
            lLambdaVolume = 10
            if lCellTypeVolume >= 100:
                lLambdaVolume = 10
            elif lCellTypeVolume >= 50:
                lLambdaVolume = 20
            elif lCellTypeVolume >= 10:
                lLambdaVolume = 30
            else:
                lLambdaVolume = 50
            lHelperXMLFileOutputStream << "    <VolumeEnergyParameters CellType=\"%s\" TargetVolume=\"%s\" LambdaVolume=\"%s\"/>\n"  % \
                (lCellTypeName, str(lCellTypeVolume), str(lLambdaVolume) )
            #lHelperXMLFileOutputStream << "    <VolumeEnergyParameters CellType="NonCondensing" TargetVolume="50" LambdaVolume="2"/>\n"
        lHelperXMLFileOutputStream << " </Plugin>\n"
        lHelperXMLFileOutputStream << "\n"

        # write to the output stream one entry per cell type vs. cell type, and its contact energy:
        lHelperXMLFileOutputStream << " <Plugin Name=\"Contact\">\n"
        lHelperXMLFileOutputStream << "   <Energy Type1=\"Wall\" Type2=\"Wall\">0</Energy>\n"
        lHelperXMLFileOutputStream << "   <Energy Type1=\"Wall\" Type2=\"Medium\">0</Energy>\n"
        lHelperXMLFileOutputStream << "   <Energy Type1=\"Medium\" Type2=\"Medium\">0</Energy>\n"

        lCellTypesCouplesDict = dict()
        lKey = 0
        for i in xrange(len(lUsedCellTypeVolumesDict)):
            # get the Type1 cell type name for the Energy definition:
            lRegionKey,lCellTypeName,lCellTypeVolume,lTypeID = lUsedCellTypeVolumesDict[lCellTypeKeys[i]]
            CDConstants.printOut("lRegionKey,lCellTypeName,lCellTypeVolume,lTypeID = " + \
                str(lRegionKey)+" "+str(lCellTypeName)+" "+str(lCellTypeVolume)+" "+str(lTypeID) , CDConstants.DebugAll )
            lHelperXMLFileOutputStream << "   <Energy Type1=\"Wall\" Type2=\"%s\">50</Energy>\n" % (lCellTypeName)
            lHelperXMLFileOutputStream << "   <Energy Type1=\"Medium\" Type2=\"%s\">16</Energy>\n" % (lCellTypeName)
            for j in xrange(len(lUsedCellTypeVolumesDict)):
                # get the Type2 cell type name for the Energy definition:
                lRegionKey2,lCellTypeName2,lCellTypeVolume2,lTypeID2 = lUsedCellTypeVolumesDict[lCellTypeKeys[j]]
                CDConstants.printOut("lRegionKey2,lCellTypeName2,lCellTypeVolume2,lTypeID2 =" + \
                    str(lRegionKey2)+" "+str(lCellTypeName2)+" "+str(lCellTypeVolume2)+" "+str(lTypeID2), CDConstants.DebugAll )
               
                # ensure that neither Type1, Type2 nor Type2, Type1 have already been defined:
                lAlreadyPresent = False
                lKeys = lCellTypesCouplesDict.keys()
                for k in xrange(len(lCellTypesCouplesDict)):
                    (lT1,lT2) = lCellTypesCouplesDict[lKeys[k]]
                    CDConstants.printOut( "(lT1,lT2) = lCellTypesCouplesDict[lKeys[k="+str(k)+"]] = "+str((lT1,lT2)) , CDConstants.DebugAll )
                    if ( (lT1 == lCellTypeName) and (lT2 == lCellTypeName2) ) or ( (lT2 == lCellTypeName) and  (lT1 == lCellTypeName2) ) :
                        lAlreadyPresent = True
                if lAlreadyPresent is not True:
                    lCellTypesCouplesDict[lKey] = (lCellTypeName,lCellTypeName2)
                    lKey = lKey + 1
        CDConstants.printOut( "lCellTypesCouplesDict = " + str(lCellTypesCouplesDict) , CDConstants.DebugAll )

        lKeys = lCellTypesCouplesDict.keys()
        for i in xrange(len(lCellTypesCouplesDict)):
            (lCellTypeName,lCellTypeName2) = lCellTypesCouplesDict[lKeys[i]]
            lHelperXMLFileOutputStream << "   <Energy Type1=\"%s\" Type2=\"%s\">16</Energy>\n" % (lCellTypeName,lCellTypeName2)

        lHelperXMLFileOutputStream << "   <Depth>2</Depth>\n"
        lHelperXMLFileOutputStream << " </Plugin>\n"
        lHelperXMLFileOutputStream << "\n"
        lHelperXMLFileOutputStream << " <Steppable Type=\"PIFInitializer\">\n"
        lHelperXMLFileOutputStream << "    <PIFName>Simulation/helpfile.piff</PIFName>\n"
        lHelperXMLFileOutputStream << " </Steppable>\n"


        lHelperXMLFileOutputStream << "\n"
        lHelperXMLFileOutputStream << "</CompuCell3D>\n"
        lHelperXMLFileOutputStream << "\n"

        lHelperXMLFile.close()


        # end progress bar:
        self.progressBar.setValue( len(lSceneItems) )
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)


        # ===============================================================================
        #   the old (now UNUSED) way of generating the cell seed points
        # ===============================================================================

        if False is True:

            lSceneItems = self.theGraphicsScene.items(QtCore.Qt.AscendingOrder)
   
            # set progress bar:
            self.progressBar.setRange(0, len(lSceneItems) )
            self.progressBar.setValue(0)
            # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
            #   unless we also explicitly ask Qt to process at least some events.
            QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

            CDConstants.printOut(  "", CDConstants.DebugExcessive )
            CDConstants.printOut( "computePottsModelAndSavePIF - running conversion to intermediate PIFF file...." , CDConstants.DebugExcessive )
            lCellID = 0
            lItemCounter = 0
   
            for lItem in lSceneItems:
   
                self.progressBar.setValue(lItemCounter+1)
                # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
                #   unless we also explicitly ask Qt to process at least some events.
                QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
   
                lItemsColor = lItem.brush().color().rgba()
                lItemsRasterSize = self.colorToCellSizeRegionDict[lItemsColor]
                lListOfPointsInItem = 0
               
                # map from scene correctly, i.e. using inverse scene transform
                #   for translation/rotation/scaling of the object:
                # print "lItem.sceneTransform() = ", lItem.sceneTransform()
                # print "lItem.sceneTransform().inverted() = ", lItem.sceneTransform().inverted()
                # NOTE: in PyQt, inverted() returns a tuple:
                lItemInverseTransform,lIsNotSingular = lItem.sceneTransform().inverted()
                if lIsNotSingular is False:
                    QtGui.QMessageBox.warning(self, "CellDraw", \
                        self.tr("Can't use scene item %1: singular QTransform.").arg(lItem))
                    return False
   
                lCellVolume = float(lItemsRasterSize * lItemsRasterSize)
                lBoundingRect = lItem.boundingRegion(lItem.sceneTransform()).boundingRect()
                print "lBoundingRect.width(), lBoundingRect.height() = ", lBoundingRect.width(), lBoundingRect.height()
                lRegionVolume = float(lBoundingRect.width() * lBoundingRect.height())
                lRequiredCellPoints = lRegionVolume / lCellVolume
                print "for lItem", lItemCounter, "lCellVolume =", lCellVolume, "lRegionVolume =", lRegionVolume, "lRequiredCellPoints =", lRequiredCellPoints
   
                lDone = False
                lCellPoints = []
                while not lDone:
                    lRandomPointOK = False
                    while not lRandomPointOK:
                        lRndX = random.random()
                        lRndX = lRndX * float(lSceneWidthInPixels - 1)
                        lRndY = random.random()
                        lRndY = lRndY * float(lSceneHeightInPixels - 1)
                        lPoint = lItemInverseTransform.map( QtCore.QPointF(lRndX,lRndY) )
                        if lItem.contains( lPoint ) :
                            lCellPoints.append( (lRndX,lRndY) )
                            # print "lCellPoints = ", lCellPoints
                            lRandomPointOK = True
                    if len(lCellPoints) >= lRequiredCellPoints:
                        lDone = True
   
                for (i,j) in lCellPoints :
   
                    lItemsColor = lItem.brush().color().rgba()
                    if lItemsColor in self.colorToNameRegionDict:
                        lRegionName = self.colorToNameRegionDict[lItemsColor]
                        xmin = int(i)
                        ymin = int(j)       
   
                        # ------------------------------------------------------------
                        # generate a random floating point number between 0.0 and the region's total probability:
                        lRegionKey = self.colorToKeyRegionDict[lItemsColor]
                        # print " prepare for RANDOM lRegionKey, lItemsColor=self.colorToKeyRegionDict[lItemsColor] =", lRegionKey, lItemsColor
                        lRnd = random.random()
                        # print " RANDOM RANDOM =", lRnd, lRnd * lProbabilityTotalsDict[lRegionKey]
                        lRnd = lRnd * lProbabilityTotalsDict[lRegionKey]
   
                        # loop through all cell type dicts for the current region until probability is matched:
                        lTheCellTypeName = ""
                        lRndCumulative = 0.0
                        lNumberOfCellTypes = len(self.regionsDict[lRegionKey][4])
                        for k in xrange(lNumberOfCellTypes) :
                            try:
                                lRndCumulative = lRndCumulative + self.regionsDict[lRegionKey][4][k][2]
                                # print "TRY lRndCumulative, self.regionsDict[lRegionKey][4][k][2] =", lRndCumulative, self.regionsDict[lRegionKey][4][k][2]
                            except:
                                lRndCumulative = self.regionsDict[lRegionKey][4][k][2]
                                # print "EXCEPT lRndCumulative, self.regionsDict[lRegionKey][4][k][2] =", lRndCumulative, self.regionsDict[lRegionKey][4][k][2]
                            # if the cell type's probability is matched by the random number, get the cell type name:
                            if ( (lTheCellTypeName == "") and (lRndCumulative > lRnd) ) :
                                lTheCellTypeName = self.regionsDict[lRegionKey][4][k][1]
                                # print "ASSIGN lTheCellTypeName = self.regionsDict[lRegionKey][4][k][1] =", lTheCellTypeName, self.regionsDict[lRegionKey][4][k][1]
                        # ------------------------------------------------------------
   
                        self.theRasterizedImageLabel.plotRect(lItemsColor, xmin, ymin, xmin, ymin)
                        self.theRasterizedImageLabel.update()
                        # print "%s %s %s %s %s %s 0 0\n"%(lCellID, lTheCellTypeName, xmin, xmin, ymin, ymin)
                        lOutputStream << "%s %s %s %s %s %s 0 0\n"%(lCellID, lTheCellTypeName, xmin, xmin, ymin, ymin)
                        lCellID +=1
   
                lItemCounter = lItemCounter+1
   
            # end for lItem in lSceneItems
            # ----------------------------

        # ===============================================================================
        # end of the old (now UNUSED) way of generating the cell seed points
        # ===============================================================================

        time.sleep(1.0)


        # ------------------------------------------------------------
        # (12) - computePottsModelAndSavePIF - (12)
        #
        # 2011 - Mitja: get help from CC3D:
        #
        # ------------------------------------------------------------


        # -----------------------------
        # erase any 'flag' file created by our helper CC3D steppables:
        lFlagFileName = os.path.join(lHelperOutputDirectoryCC3D,"flagfile.text")
        if os.path.isfile(lFlagFileName) :
            os.remove(lFlagFileName)

        # -----------------------------
        # run CC3D on the .piff and .xml files we just prepared:
        CUR_DIR = os.path.join(lHelperOutputDirectoryCC3D,"helpfile_CellDraw.cc3d")
        CDConstants.printOut( "=====>=====> CUR_DIR = " + str(CUR_DIR) , CDConstants.DebugVerbose )
        self.startCC3D(CUR_DIR)

        # -----------------------------
        # now check the existence of a flag file, once a second:
        lCC3DhasCompleted = False
        while lCC3DhasCompleted is not True:
            time.sleep(1.0)
            CDConstants.printOut( "trying to read file " + str(lFlagFileName) , CDConstants.DebugAll )
            if os.path.isfile(lFlagFileName) :
                try:
                    lFile = open(lFlagFileName, 'r')
                    CDConstants.printOut( "correctly opened file" + str(lFlagFileName), CDConstants.DebugVerbose )
                    lFlagFileContent = lFile.readline()
                    if lFlagFileContent == str("PIFF output from CC3D done.\n"):
                        CDConstants.printOut( "file "+str(lFlagFileName)+" says ... "+str(lFlagFileContent)+" ...yay!", CDConstants.DebugVerbose )
                        lCC3DhasCompleted = True
                    else:
                        CDConstants.printOut( "file "+str(lFlagFileName)+" says ... "+str(lFlagFileContent)+" ...why?" , CDConstants.DebugVerbose )
                    lFile.close()
                except:
                    CDConstants.printOut( " _____________ in cdSceneRasterizer: can not read from file "+str(lFlagFileName)+" _____________" , CDConstants.DebugVerbose )
                    QtGui.QMessageBox.warning(self, "CellDraw", \
                        self.tr("Cannot write file %1 .\n[in computePottsModelAndSavePIF() - (12a)]").arg(lFlagFileName) )
                    self.hide()
                    return False
               
        # -----------------------------
        if os.path.isfile(lFlagFileName) :
            os.remove(lFlagFileName)


        # -----------------------------
        # get the intermediate PIFF file prepared by CC3D and convert it:
        lCC3DGeneratedPIFFileName = os.path.join(lHelperOutputDirectoryCC3D,"helpfileoutputfrompotts.piff")

        # 2010 - Mitja: load the file's data into a QImage object:
        lCC3DGeneratedPIFFile = QtCore.QFile(lCC3DGeneratedPIFFileName)
        lFileOK = lCC3DGeneratedPIFFile.open(QtCore.QIODevice.ReadOnly)
        if lFileOK is False:
            CDConstants.printOut( " _____________ in cdSceneRasterizer: computePottsModelAndSavePIF() - (12a) - can not read from file " + \
                str(lCC3DGeneratedPIFFileName) +" _____________" , CDConstants.DebugImportant )
            QtGui.QMessageBox.warning( self, self.tr("CellDraw"), \
                self.tr("Cannot read from file %1 .\n[in computePottsModelAndSavePIF() - (12a)]").arg(lCC3DGeneratedPIFFileName) )
            self.hide()
            return False
        else:
            # i.e. the lCC3DGeneratedPIFFile has been opened fine for reading:
            CDConstants.printOut( " _____________ in cdSceneRasterizer: computePottsModelAndSavePIF() - (12a) - correctly opened file " + \
                str(lCC3DGeneratedPIFFileName) +" _____________" , CDConstants.DebugVerbose )

            #self.infoLabel.setText("123456789012345678901234567890123456789012345678901234567890")
            self.infoLabel.setText( self.tr("Reading Potts data from CC3D in file: %1").arg(lCC3DGeneratedPIFFileName) )


        # convert the CC3D-generated intermediate PIFF file's line endings, so that it's OK on any platform:
        lThePIFText = QtCore.QTextStream(lCC3DGeneratedPIFFile).readAll()
        lThePIFText.replace("\r\n", "\n")
        lThePIFText.replace("\r", "\n")
        lThePIFTextList = lThePIFText.split("\n")
        CDConstants.printOut( " " , CDConstants.DebugAll )
        CDConstants.printOut( "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=" , CDConstants.DebugAll )
        CDConstants.printOut( "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=" , CDConstants.DebugAll )
        CDConstants.printOut( str(lThePIFTextList) , CDConstants.DebugAll )
        CDConstants.printOut( "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=" , CDConstants.DebugAll )
        CDConstants.printOut( "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=" , CDConstants.DebugAll )
        CDConstants.printOut( " " , CDConstants.DebugAll )
        # close the intermediate PIFF file from CC3D we had used for reading:
        lCC3DGeneratedPIFFile.close()
        # delete the intermediate PIFF file from CC3D we had used for reading:
        if os.path.isfile(lCC3DGeneratedPIFFileName) :
            os.remove(lCC3DGeneratedPIFFileName)

        self.infoLabel.setText( "Received intermediate Potts data from CC3D." )

        # ------------------------------------------------------------
        # (13) - computePottsModelAndSavePIF - (13)
        #
        #       finally set up saving to the final proper PIFF file,
        #       converting from the CC3D-generated intermediate PIFF file:
        #
        # ------------------------------------------------------------


        # provide user feedback, reusing the lTmpPixmap and lTmpPainter used above:
        lTmpPixmap.fill(QtCore.Qt.transparent)
        lTmpPainter = QtGui.QPainter(lTmpPixmap)
        lTmpColor = QtGui.QColor()
        lTmpPen = QtGui.QPen()
        lTmpPen.setWidth(1)
        lTmpPen.setCosmetic(True)

        # line by line, read from the intermediate PIFF information's line endings, so that it's OK on any platform:
        for lThePIFTextLine in lThePIFTextList:
            # if we have a lThePIFTextLine, it doesn't necessarily follow
            #   that it's a well-formed PIFF line... so we better use "try - except" :
            try:                   
                # print "lThePIFTextLine", i, "is <",  lThePIFTextLine, ">"
                thePIFlineList = lThePIFTextLine.split(" ", QtCore.QString.SkipEmptyParts)
                pifCellID = int(thePIFlineList[0])
                pifCellTypeID = int(thePIFlineList[1])
                pifXMin = int(thePIFlineList[2])
                pifXMax = int(thePIFlineList[3])
                pifYMin = int(thePIFlineList[4])
                pifYMax = int(thePIFlineList[5])
                pifZMin = int(thePIFlineList[6])
                pifZMax = int(thePIFlineList[7])
                # since the intermediate PIFF file generated by CC3D doesn't contain type names, retrieve them from our dict:
                lTheCellTypeName = lUsedCellIDToNameDict[pifCellTypeID]
                lTheCellTypeColor = lUsedCellIDToColorDict[pifCellTypeID]
                # provide user feedback:
                CDConstants.printOut( "-=- " + \
                    str(lTheCellTypeColor)+" "+str(pifXMin)+" "+str(pifXMax)+" "+str(pifYMin)+" "+str(pifYMax)+ " -=-", \
                    CDConstants.DebugAll )
#                 TODO TODO TODO fix why the min-max produce huge rectangles here !!! TODO TODO TODO   
                lTmpColor.setRgba( lTheCellTypeColor.rgba() )
                lTmpPen.setColor(lTmpColor)
                lTmpPainter.setPen(lTmpPen)
                lTmpPainter.drawRect(pifXMin, pifYMin, pifXMax-pifXMin, pifYMax-pifYMin)
            except:
                # we got exception in parsing a PIFF line, just do nothing.
                pass
        lTmpPainter.end()
        self.theRasterizedImageLabel.drawPixmapAtPoint(lTmpPixmap)
        self.theRasterizedImageLabel.update()



        lToBeSavedFileExtension = QtCore.QString("piff")
        lToBeSavedInitialPath = QtCore.QDir.currentPath() + self.tr("/untitled.") + lToBeSavedFileExtension
        lFileName = QtGui.QFileDialog.getSaveFileName(self, self.tr("CellDraw - Save PIFF file from Potts algorithm as"),
                               lToBeSavedInitialPath,
                               self.tr("%1 files (*.%2);;All files (*)")
                                   .arg(lToBeSavedFileExtension.toUpper())
                                   .arg(lToBeSavedFileExtension))
        if lFileName.isEmpty():
            CDConstants.printOut( "___ - DEBUG ----- PIFInputMainWindow: fileSavePIFFromScene_Callback() Potts-generated PIFF failed: no filename selected.", \
                CDConstants.DebugAll )
            self.hide()
            return False


        # open output file, and make sure that it's writable:
        lFile = QtCore.QFile(lFileName)
        lOnlyThePathName,lOnlyTheFileName = os.path.split(str(lFileName))
        #self.infoLabel.setText("123456789012345678901234567890123456789012345678901234567890")
        self.infoLabel.setText( self.tr("Saving final PIFF from CC3D's Potts to file: %1").arg(lOnlyTheFileName) )
        if not lFile.open( QtCore.QFile.WriteOnly | QtCore.QFile.Text):
            QtGui.QMessageBox.warning(self, "CellDraw", \
                    self.tr("Cannot write file %1 .\nError: [%2] .").arg(lOnlyTheFileName).arg(lFile.errorString()))
            self.hide()
            return False

        # open a QTextStream, i.e. an "interface for reading and writing text":
        lOutputStream = QtCore.QTextStream(lFile)

        # line by line, read from the intermediate PIFF information's line endings, so that it's OK on any platform:
        for lThePIFTextLine in lThePIFTextList:
            # if we have a lThePIFTextLine, it doesn't necessarily follow
            #   that it's a well-formed PIFF line... so we better use "try - except" :
            try:                   
                # print "lThePIFTextLine", i, "is <",  lThePIFTextLine, ">"
                thePIFlineList = lThePIFTextLine.split(" ", QtCore.QString.SkipEmptyParts)
                pifCellID = int(thePIFlineList[0])
                pifCellTypeID = int(thePIFlineList[1])
                pifXMin = int(thePIFlineList[2])
                pifXMax = int(thePIFlineList[3])
                pifYMin = int(thePIFlineList[4])
                pifYMax = int(thePIFlineList[5])
                pifZMin = int(thePIFlineList[6])
                pifZMax = int(thePIFlineList[7])
                # since the intermediate PIFF file generated by CC3D doesn't contain type names, retrieve them from our dict:
                lTheCellTypeName = lUsedCellIDToNameDict[pifCellTypeID]
                lOutputStream << "%s %s %s %s %s %s %s %s\n"%(pifCellID, lTheCellTypeName, pifXMin, pifXMax, pifYMin, pifYMax, pifZMin, pifZMax)
            except:
                # we got exception in parsing a PIFF line, just do nothing.
                pass

        # cleanly close access to the file:
        lFile.close()

        #self.infoLabel.setText("123456789012345678901234567890123456789012345678901234567890")
        self.infoLabel.setText( self.tr("Saved PIFF from CC3D Potts to file %1 complete.").arg(lOnlyTheFileName) )

        CDConstants.printOut( "computePottsModelAndSavePIF():                       PIFF file saving from Potts complete.\n" , CDConstants.DebugExcessive )

        self.hide()


    # end of def computePottsModelAndSavePIF(self)
    # ---------------------------------------------------------












    # ---------------------------------------------------------
    def createProgressBar(self):
        self.progressBar = QtGui.QProgressBar()
        self.progressBar.setRange(0, 10000)
        self.progressBar.setValue(0)
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    # ---------------------------------------------------------
    def advanceProgressBar(self):
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        self.progressBar.setValue(curVal + (maxVal - curVal) / 100)
        # Qt/PyQt's progressBar won't display updates from setValue(...) calls,
        #   unless we also explicitly ask Qt to process at least some events.
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    # ---------------------------------------------------------







# end of class CDSceneRasterizer(QtGui.QWidget)
# ======================================================================












# ======================================================================
# the following if statement checks whether the present file
#    is currently being used as standalone (main) program, and in this
#    class's (CDSceneRasterizer) case it is simply used for testing:
# ======================================================================
if __name__ == '__main__':


    pogi = pigi
    # call them!
    gigi()
    pigi()
    pogi()

    CDConstants.printOut( "001 - DEBUG - mi __main__ xe 01" , CDConstants.DebugExcessive )
    # every PyQt4 app must create an application object, from the QtGui module:
    miApp = QtGui.QApplication(sys.argv)

    CDConstants.printOut( "002 - DEBUG - mi __main__ xe 02" , CDConstants.DebugExcessive )

    # the window containing the rasterized image:
    mainPanel = CDSceneRasterizer()

    CDConstants.printOut( "003 - DEBUG - mi __main__ xe 03" , CDConstants.DebugExcessive )

    miBoringGraphicsScene = QtGui.QGraphicsScene()

    # temporarily assign colors/names to a dictionary:
    #   place all used colors/type names into a dictionary, to be locally accessible
    miDict = dict({ 1: [ QtGui.QColor(255,0,0), "RedTypeName", 10, 0, \
                         [[QtGui.QColor(255,128,128), "redCondensing", 0.5], \
                          [QtGui.QColor(255,128,128), "redNonCondensing", 0.5]] ], \
                    2: [QtGui.QColor(0,255,0), "GreenTypeName", 10, 0, \
                         [[QtGui.QColor(128,255,128), "greenCondensing", 0.5], \
                          [QtGui.QColor(128,255,128), "greenNonCondensing", 0.5]] ], \
                    3: [QtGui.QColor(0,0,255), "BlueTypeName", 10, 0, \
                         [[QtGui.QColor(128,128,255), "blueCondensing", 0.5], \
                          [QtGui.QColor(128,128,255), "blueNonCondensing", 0.5]] ], \
                    4: [QtGui.QColor(255,255,0), "YellowTypeName", 10, 0, \
                         [[QtGui.QColor(255,255,128), "yellowCondensing", 0.5], \
                          [QtGui.QColor(255,255,128), "yellowNonCondensing", 0.5]] ], \
                    5: [QtGui.QColor(255,0,255), "PurpleTypeName", 10, 0, \
                         [[QtGui.QColor(255,128,255), "purpleCondensing", 0.5], \
                          [QtGui.QColor(255,128,255), "purpleNonCondensing", 0.5]] ]     } )

    mainPanel.setInputGraphicsScene(miBoringGraphicsScene)

    mainPanel.setRegionsDict(miDict)
    # mainPanel.populateTableWithRegionsDict()

    # show() and raise_() have to be called here:
    mainPanel.show()
    # raise_() is a necessary workaround to a PyQt-caused (or Qt-caused?) bug on Mac OS X:
    #   unless raise_() is called right after show(), the window/panel/etc will NOT become
    #   the foreground window and won't receive user input focus:
    mainPanel.raise_()
    CDConstants.printOut( "004 - DEBUG - mi __main__ xe 04" , CDConstants.DebugExcessive )

    sys.exit(miApp.exec_())
    CDConstants.printOut( "005 - DEBUG - mi __main__ xe 05" , CDConstants.DebugExcessive )

# end if __name__ == '__main__'


# Local Variables:
# coding: US-ASCII
# End:
