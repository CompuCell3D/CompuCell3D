#!/usr/bin/env python

from PyQt4 import QtCore, QtGui


from PyQt4 import QtGui # from PyQt4.QtGui import *
from PyQt4 import QtCore # from PyQt4.QtCore import *

import math   # we need to import math to use sqrt() and such
import sys    # for get/setrecursionlimit()


# 2010 - Mitja:
import inspect # for debugging functions, remove in final version
# debugging functions, remove in final version
def debugWhoIsTheRunningFunction():
    return inspect.stack()[1][3]
def debugWhoIsTheParentFunction():
    return inspect.stack()[2][3]


# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ----------------------------------------------------------------------
# 2011- Mitja: additional layer to draw an input image for CellDraw
#   on the top of the QGraphicsScene-based cell/region editor
# ------------------------------------------------------------
# This class draws an image loaded from a file,
#   and it processes mouse click events.
# ------------------------------------------------------------
class CDImageLayer(QtCore.QObject):
    # --------------------------------------------------------

    # we pass a "dict" parameter with the signalThatMouseMoved parameter, so that
    #   both the mouse x,y coordinates as well as color information can be passed around:

    signalThatMouseMoved = QtCore.pyqtSignal(dict)
    
#     self.scene.signalThatSceneResized.connect(self.handlerForSceneResized)

    # 2010 - Mitja: add functionality for drawing color regions:
    #    where the value for keeping track of what's been changed is:
    #    0 = Color Pick = CDConstants.ImageModePickColor
    #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
    #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon

    # --------------------------------------------------------
    def __init__(self, pParent=None):
        print "|^|^|^|^| CDImageLayer.__init__( pParent ==", pParent, ") |^|^|^|^|"

        QtCore.QObject.__init__(self, pParent)

        # the class global keeping track of the required opacity:
        #    0.0 = minimum = the image is completely transparent (invisible)
        #    1.0 = minimum = the image is completely opaque
        self.imageOpacity = 1.0
       
        # the class global keeping track of the fuzzy pick treshold:
        #    0.0 = minimum = pick only the seed color
        #    1.0 = minimum = pick everything in the image
        self.fuzzyPickTreshold = 0.02

        self.width = 240
        self.height = 180
        self.pifSceneWidth = self.width
        self.pifSceneHeight = self.height
        self.fixedRasterWidth = 10
        # the "theImage" QImage is the original image loaded from a picture file:
        self.theImage = QtGui.QImage()
        # the "processedImage" QImage is the one we process to pick from:
        self.processedImage = QtGui.QImage()
        # the scale factor is the zoom factor for viewing the image, separate from the PIFF scene zoom:
        self.scaleFactor = 1.0
#
#         # the "thePixmap" is to mimic a QLabel which can be assigned a QPixmap:
#         self.thePixmap = QtGui.QPixmap()
#         # the "processedPixmap" QPixmap is the one we obtained from processedImage:
#         self.processedPixmap = QtGui.QPixmap()

        if isinstance( pParent, QtGui.QWidget ) is True:
            self.graphicsSceneWindow = pParent
        else:
            self.graphicsSceneWindow = None
        print "|^|^|^|^| CDImageLayer.__init__() self.graphicsSceneWindow ==", \
            self.graphicsSceneWindow, " |^|^|^|^|"

        self.inputImagePickingMode = CDConstants.ImageModeDrawFreehand

        #    highlighting flag, used to provide feedback to the user
        #    for when the mouse position is close to its starting point:
        self.myPathHighlight = False

        # this image's list of vertices for painter path:
        self.theCurrentPath = []

        # debugging counter, how many times has the paint function been called:
        self.repaintEventsCounter = 0

        # mouse handling:
        self.myMouseX = -1
        self.myMouseY = -1
        self.myMouseDownX = -1
        self.myMouseDownY = -1
        # "Old" variables are handy to keep previous mouse position value:
        self.myMouseXOld = -1
        self.myMouseYOld = -1
        # you could compute the mouse movement's speed:
        self.myMouseSpeed = 0.0
        # mouse button state, we set it to True if the Left Mouse button is pressed:
        self.myMouseLeftDown = False

        # 2010 - Mitja: add flag to show that CDImageLayer
        #   holds an image and pixmap as loaded from a file.
        self.imageLoadedFromFile = False



    # ------------------------------------------------------------------
    # 2011 - Mitja: setScaleZoom() is to set the image display scale/zoom factor:
    # ------------------------------------------------------------------   
    def setScaleZoom(self, pValue):
        self.scaleFactor = pValue
        self.setToProcessedImage()


    # ------------------------------------------------------------------
    # 2011 - Mitja: setImageOpacity() is to set input image opacity:
    # ------------------------------------------------------------------   
    def setImageOpacity(self, pValue):
        # the class global keeping track of the required opacity:
        #    0.0 = minimum = the image is completely transparent (invisible)
        #    1.0 = minimum = the image is completely opaque
        self.imageOpacity = 0.01 * (pValue)
        # 2011 - Mitja: update the CDImageLayer's parent widget,
        #   i.e. paintEvent() will be invoked regardless of the picking mode:
        # but we don't call update() here, we call it where opacity is changed
        # self.graphicsSceneWindow.scene.update()


    # ------------------------------------------------------------------
    # 2011 - Mitja: setImageOpacity() is to set :
    # ------------------------------------------------------------------   
    def setFuzzyPickTreshold(self, pValue):
        # the class global keeping track of the fuzzy pick treshold:
        #    0.0 = minimum = pick only the seed color
        #    1.0 = minimum = pick everything in the image
        self.fuzzyPickTreshold = (0.01 * float(pValue)) * (0.01 * float(pValue))


    # ------------------------------------------------------------------
    # 2011 - Mitja: the setImageLoadedFromFile() function is to mimic a QLabel,
    #   which can be assigned a QPixmap - there's no real need to have both
    #   a QPixmap and a QImage in the end, so TODO remove one of the two:
    # ------------------------------------------------------------------   
    def setImageLoadedFromFile(self, pTrueOrFalse):
        self.imageLoadedFromFile = pTrueOrFalse
        if isinstance( self.graphicsSceneWindow, QtGui.QWidget ) == True:
            self.graphicsSceneWindow.scene.update()


    # ------------------------------------------------------------------
    # 2011 - Mitja: the setMouseTracking() function is to mimic a QLabel,
    #   which being a QWidget supports setMouseTracking(). Instead, we
    #   pass it upstream to the parent widget's QGraphicsScene:
    # ------------------------------------------------------------------   
    def setMouseTracking(self, pTrueOrFalse):
        if isinstance( self.graphicsSceneWindow, QtGui.QWidget ) == True:
#             print "self.graphicsSceneWindow.view.hasMouseTracking() ==============", \
#               self.graphicsSceneWindow.view.hasMouseTracking()
#             print "                                    pTrueOrFalse ==============", \
#               pTrueOrFalse
            self.graphicsSceneWindow.view.setMouseTracking(pTrueOrFalse)
#             print "self.graphicsSceneWindow.view.hasMouseTracking() ==============", \
#               self.graphicsSceneWindow.view.hasMouseTracking()



    # ------------------------------------------------------------------
    # 2011 - Mitja: the setPixmap() function is to mimic a QLabel,
    #   which can be assigned a QPixmap - there's no real need to have both
    #   a QPixmap and a QImage in the end, so TODO remove one of the two:
    # ------------------------------------------------------------------   
#     def setPixmap(self, pPixmap):
#         if isinstance(pPixmap, QtGui.QPixmap):
#             self.thePixmap = QtGui.QPixmap(pPixmap)
#             self.processedPixmap = QtGui.QPixmap(pPixmap)




    # ------------------------------------------------------------------
    # 2011 - Mitja: the setWidthOfFixedRaster() function is to set the fixed raster width:
    # ------------------------------------------------------------------   
    def setWidthOfFixedRaster(self, pGridWidth):
        self.fixedRasterWidth = pGridWidth


    # ------------------------------------------------------------------
    # 2011 - Mitja: the setImage() function is to assign the starting QImage:
    # ------------------------------------------------------------------   
    def setImage(self, pImage):
        if isinstance(pImage, QtGui.QImage):

            if self.imageLoadedFromFile == True:       
                # immediately transform the starting/loaded QImage: invert Y values, from RHS to LHS:
                lWidth = pImage.width()
                lHeight = pImage.height()
   
                # create an empty pixmap where to store the composed image:
                lResultPixmap = QtGui.QPixmap(lWidth, lHeight)
   
                # create a QPainter to perform the Y inversion operation:
                lPainter = QtGui.QPainter(lResultPixmap)
                lPainter.fillRect(lResultPixmap.rect(), QtCore.Qt.transparent)
   
                # take care of the RHS <-> LHS mismatch at its visible end,
                #   by flipping the y coordinate in the QPainter's affine transformations:       
                lPainter.translate(0.0, float(lHeight))
                lPainter.scale(1.0, -1.0)
       
                # access the QLabel's pixmap to draw it explicitly, using QPainter's scaling:
                lPainter.drawPixmap(QtCore.QPoint(0,0), QtGui.QPixmap.fromImage(pImage))
   
                lPainter.end()           
   
                # copy the QImage passed as parameter into two separate instances:
                self.theImage = QtGui.QImage(lResultPixmap.toImage())
                # instead of setting manually, we call:     # self.processedImage = QtGui.QImage(lResultPixmap.toImage())
                self.setToProcessedImage()

            else:
                # directly copy the QImage passed as parameter into two separate instances:
                self.theImage = QtGui.QImage(pImage)
                # instead of setting manually, we call:    # self.processedImage = QtGui.QImage(pImage)
                self.setToProcessedImage()

            self.width = self.processedImage.width()
            self.height = self.processedImage.height()


    # ------------------------------------------------------------------
    # 2011 - Mitja: the setToProcessedImage() function restores the original
    #   QImage loaded from a file (i.e. self.theImage) back into the
    #   processed QImage (i.e. self.processedImage) undoing all color processing.
    # ------------------------------------------------------------------   
    def setToProcessedImage(self):
        if isinstance(self.theImage, QtGui.QImage):
            # copy self.theImage into a separate instance for processing:
            if (self.scaleFactor >= 1.001) or (self.scaleFactor <= 0.999) :
                self.processedImage = QtGui.QImage(self.theImage).scaled( \
                    int( float(self.processedImage.width()) * self.scaleFactor ), \
                    int( float(self.processedImage.height()) * self.scaleFactor ), \
                    aspectRatioMode = QtCore.Qt.KeepAspectRatio, \
                    transformMode = QtCore.Qt.SmoothTransformation  )
            else:
                self.processedImage = QtGui.QImage(self.theImage)

        else:
            self.processedImage = QtGui.QImage()
            print "CDImageLayer.setToProcessedImage() ERROR: no self.theImage!"


    # ------------------------------------------------------------
    # 2011 - Mitja: provide color-to-color distance calculation:
    # ------------------------------------------------------------
    def colorToColorIsCloseDistance(self, pC1, pC2, pDist):
        r1 = QtGui.QColor(pC1).redF()
        g1 = QtGui.QColor(pC1).greenF()
        b1 = QtGui.QColor(pC1).blueF()
        r2 = QtGui.QColor(pC2).redF()
        g2 = QtGui.QColor(pC2).greenF()
        b2 = QtGui.QColor(pC2).blueF()
       
        d =   ((r2-r1)*0.30) * ((r2-r1)*0.30) \
            + ((g2-g1)*0.59) * ((g2-g1)*0.59) \
            + ((b2-b1)*0.11) * ((b2-b1)*0.11)
        if (d < pDist) :
            retVal = True
        else:
            retVal = False
        # print "r1, g1, b1, r2, g2, b2, d, pDist =", r1, g1, b1, r2, g2, b2, "|", d, pDist, retVal
        return retVal


    # ------------------------------------------------------------
    # 2011 - Mitja: provide color-to-color distance calculation:
    # ------------------------------------------------------------
    def colorToColorIsSame(self, pC1, pC2):
        r1 = QtGui.QColor(pC1).redF()
        g1 = QtGui.QColor(pC1).greenF()
        b1 = QtGui.QColor(pC1).blueF()
        r2 = QtGui.QColor(pC2).redF()
        g2 = QtGui.QColor(pC2).greenF()
        b2 = QtGui.QColor(pC2).blueF()
       
        dr = (r2-r1) * (r2-r1)
        dg = (g2-g1) * (g2-g1)
        db = (b2-b1) * (b2-b1)
        if (dr < 0.0001) and (dg < 0.0001) and (db < 0.0001) :
#             print "TRUE",  r1, r2, g1, g2, b1, b2
            return True
        else:
#             print "FALSE",  r1, r2, g1, g2, b1, b2
            return False


    # ------------------------------------------------------------
    # 2011 - Mitja: compute the image with flattened all colors "close" to the one picked:
    #  NOTE: this is not "magic wand" style flood-fill picking, but rather flattening
    #        of all the close colors over the *entire* image!
    # ------------------------------------------------------------
    def processImageForCloseColors(self, pX, pY):
        # create a copy of the image:
        self.processedImage = QtGui.QImage(self.theImage)

        lWidth = self.processedImage.width()
        lHeight = self.processedImage.height()
        lSeedColor = self.theImage.pixel(pX, pY)
        print "01"
        # 2010 - Mitja: python's xrange function is more appropriate for large loops
        #   since it generates integers (the range function generates lists instead)
        for i in xrange(0, lWidth, 1):
            for j in xrange(0, lHeight, 1):
                lTmpColor = self.theImage.pixel(i,j)
                if self.colorToColorIsCloseDistance(lSeedColor, lTmpColor, self.fuzzyPickTreshold) :
                    lTmpColor = lSeedColor
                    self.processedImage.setPixel(i,j,lTmpColor)

#         self.processedPixmap = QtGui.QPixmap(self.processedImage)
        print "10"
        # self.theImage = QtGui.QImage(self.processedImage)


    # ------------------------------------------------------------
    # 2011 - Mitja: compute the image with fuzzy pick of all colors that
    #   are "close" to the one picked, the so-called "magic wand"
    # ------------------------------------------------------------
    def processImageForFuzzyPick(self, pX, pY):
        lWidth = self.processedImage.width()
        lHeight = self.processedImage.height()
        lSeedColor = self.theImage.pixel(pX, pY)
        # create a blank image:
        # self.processedImage = QtGui.QImage(self.theImage.width(), self.theImage.height(), self.theImage.format())

        # create a copy image:
        self.processedImage = QtGui.QImage(self.theImage)


#         lSysRecursionLimit = sys.getrecursionlimit()
#         print lSysRecursionLimit
#         sys.setrecursionlimit(100000000)
#         print sys.getrecursionlimit()
        print "02"
        self.floodFillFuzzyPick(pX, pY, lSeedColor)
#         self.processedPixmap = QtGui.QPixmap(self.processedImage)
        print "20"
#         sys.setrecursionlimit(lSysRecursionLimit)
#         print lSysRecursionLimit
#         # self.theImage = QtGui.QImage(self.processedImage)

#     def floodFill(self, pX, pY, pOld, pNew):
#         lTmpColor = self.theImage.pixel(pX, pY)
#         if self.colorToColorIsSame(pOld, lTmpColor) :
#             self.processedImage.setPixel(pX, pY, pNew)
#             self.floodFill(pX - 1,     pY, pOld, pNew)
#             self.floodFill(pX + 1,     pY, pOld, pNew)
#             self.floodFill(    pX, pY - 1, pOld, pNew)
#             self.floodFill(    pX, pY + 1, pOld, pNew)


    # 2011 - Mitja: fuzzy pick uses the flood-fill algorithm as
    # inspired by ESR's code at:
    # http://www.mail-archive.com/image-sig@python.org/msg00489.html

    def floodFillFuzzyPick(self, pX, pY, pReplacementColor):
        lWidth = self.processedImage.width()
        lHeight = self.processedImage.height()
        print "Flood fill on a region of non-BORDER_COLOR pixels."
        edge = [(pX, pY)]
        self.processedImage.setPixel(pX, pY, pReplacementColor)
        while edge:
            newedge = []
            for (pX, pY) in edge:
                for (s, t) in ((pX+1, pY), (pX-1, pY), (pX, pY+1), (pX, pY-1)):
                    if (s >= 0) and (s < lWidth) and (t >= 0) and (t < lHeight) :
                        lColorAtST = self.processedImage.pixel(s, t)
                        if (self.colorToColorIsSame( pReplacementColor, lColorAtST) != True) and \
                                (self.colorToColorIsCloseDistance( \
                                    pReplacementColor, lColorAtST, self.fuzzyPickTreshold) \
                                == True):
                            self.processedImage.setPixel(s, t, pReplacementColor)
                            newedge.append((s, t))
            edge = newedge
#
#
#     def __flood_fill(self, image, x, y, value):
#         "Flood fill on a region of non-BORDER_COLOR pixels."
#         if not image.within(x, y) or image.get_pixel(x, y) == BORDER_COLOR:
#             return
#         edge = [(x, y)]
#         image.set_pixel(x, y, value)
#         while edge:
#             newedge = []
#             for (x, y) in edge:
#                 for (s, t) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
#                     if image.within(s, t) and \
#                         image.get_pixel(s, t) not in (BORDER_COLOR, value):
#                         image.set_pixel(s, t, value)
#                         newedge.append((s, t))
#             edge = newedge

#
#     def paintFloodFill ( targetImage, fillImage, point, targetColour, replacementColour, tolerance, extendFillImage) {
#    
#     queue = QtCore.Qt.QList()  # QPoints queue for all the pixels of the filled area (as they are found)
#     int j, k; bool condition;
#     replaceImage = QImage
#     
#     
#
#         targetImage->extend(fillImage->boundaries); // not necessary - here just to prevent some bug when we draw outside the targetImage - to be fixed
#         replaceImage = new BitmapImage(NULL, fillImage->boundaries, QtGui.QColor(0,0,0,0));
#         replaceImage->extendable = false;
#     }
#     //QPainter painter1(replaceImage->image);
#     //QPainter painter2(fillImage->image);
#     //painter1.setPen( QtGui.QColor(replacementColour) );
#     QPen myPen;
#     myPen = QPen( QtGui.QColor(replacementColour) , 1.0, Qt::SolidLine, Qt::RoundCap,Qt::RoundJoin);
#
#     targetColour = targetImage->pixel(point.x(), point.y());
#     //if(  rgbDistance(targetImage->pixel(point.x(), point.y()), targetColour) > tolerance ) return;
#     queue.append( point );
#     // ----- flood fill
#     // ----- from the standard flood fill algorithm
#     // ----- http://en.wikipedia.org/wiki/Flood_fill
#     j = -1; k = 1;
#     for(int i=0; i< queue.size(); i++ ) {
#         point = queue.at(i);
#         if(  replaceImage->pixel(point.x(), point.y()) != replacementColour  && rgbDistance(targetImage->pixel(point.x(), point.y()), targetColour) < tolerance ) {
#             j = -1; condition =  (point.x() + j > targetImage->left());
#             if(!extendFillImage) condition = condition && (point.x() + j > replaceImage->left());
#             while( replaceImage->pixel(point.x()+j, point.y()) != replacementColour  && rgbDistance(targetImage->pixel( point.x()+j, point.y() ), targetColour) < tolerance && condition) {
#                 j = j - 1;
#                 condition =  (point.x() + j > targetImage->left());
#                 if(!extendFillImage) condition = condition && (point.x() + j > replaceImage->left());
#             }
#
#             k = 1; condition = ( point.x() + k < targetImage->right()-1);
#             if(!extendFillImage) condition = condition && (point.x() + k < replaceImage->right()-1);
#             while( replaceImage->pixel(point.x()+k, point.y()) != replacementColour  && rgbDistance(targetImage->pixel( point.x()+k, point.y() ), targetColour) < tolerance && condition) {
#                 k = k + 1;
#                 condition = ( point.x() + k < targetImage->right()-1);
#                 if(!extendFillImage) condition = condition && (point.x() + k < replaceImage->right()-1);
#             }
#
#             //painter1.drawLine( point.x()+j, point.y(), point.x()+k+1, point.y() );
#
#             replaceImage->drawLine( QPointF(point.x()+j, point.y()), QPointF(point.x()+k, point.y()), myPen, QPainter::CompositionMode_SourceOver, false);
#             //for(int l=0; l<=k-j+1 ; l++) {
#             //    replaceImage->setPixel( point.x()+j, point.y(), replacementColour );
#             //}
#
#             for(int x = j+1; x < k; x++) {
#                 //replaceImage->setPixel( point.x()+x, point.y(), replacementColour);
#                 condition = point.y() - 1 > targetImage->top();
#                 if(!extendFillImage) condition = condition && (point.y() - 1 > replaceImage->top());
#                 if( condition && queue.size() < targetImage->height() * targetImage->width() ) {
#                     if( replaceImage->pixel(point.x()+x, point.y()-1) != replacementColour) {
#                         if(rgbDistance(targetImage->pixel( point.x()+x, point.y() - 1), targetColour) < tolerance) {
#                             queue.append( point + QPoint(x,-1) );
#                         } else {
#                             replaceImage->setPixel( point.x()+x, point.y()-1, replacementColour);
#                         }
#                     }
#                 }
#                 condition = point.y() + 1 < targetImage->bottom();
#                 if(!extendFillImage) condition = condition && (point.y() + 1 < replaceImage->bottom());
#                 if( condition && queue.size() < targetImage->height() * targetImage->width() ) {
#                     if( replaceImage->pixel(point.x()+x, point.y()+1) != replacementColour) {
#                         if(rgbDistance(targetImage->pixel( point.x()+x, point.y() + 1), targetColour) < tolerance) {
#                             queue.append( point + QPoint(x, 1) );
#                         } else {
#                             replaceImage->setPixel( point.x()+x, point.y()+1, replacementColour);
#                         }
#                     }
#                 }
#             }
#         }
#     }        
#     //painter2.drawImage( QPoint(0,0), *replaceImage );
#     //bool memo = fillImage->extendable;
#     //fillImage->extendable = false;
#     fillImage->paste(replaceImage);
#     //fillImage->extendable = memo;
#     //replaceImage->fill(qRgba(0,0,0,0));
#     //painter1.end();
#     //painter2.end();
#     delete replaceImage;
#     //update();
#     
#




    # ------------------------------------------------------------------
    # 2011 - Mitja: paintTheImageLayer() paints/draws all that needs to go
    #   into the Image Layer, and may be called directly or by our paintEvent() handler:
    # ------------------------------------------------------------------   
    def paintTheImageLayer(self, pThePainter):

        CDConstants.printOut("[F] hello, I'm "+str(debugWhoIsTheRunningFunction())+", parent is "+str(debugWhoIsTheParentFunction())+ \
            " ||||| self.repaintEventsCounter=="+str(self.repaintEventsCounter)+ \
            " ||||| CDImageLayer.paintTheImageLayer(pThePainter=="+str(pThePainter)+")", CDConstants.DebugTODO )

        # paint into the passed QPainter parameter:
        lPainter = pThePainter

        # draw the input image, if there is one:
        if isinstance( self.processedImage, QtGui.QImage ) == True:
            lPixMap = QtGui.QPixmap.fromImage(self.processedImage)

#             lTmpOpacity = lPainter.opacity()

#             if self.inputImagePickingMode is CDConstants.ImageModePickColor:
#                 # if we are in color picking mode, draw the image fully opaque:
#                 lPainter.setOpacity(1.0)
#             else:
#                 # for freehand and polygon drawing mode, draw the image translucent:
            # push the QPainter's current state onto a stack,
            #   to be followed by a restore() below:
            lPainter.save()

            lPainter.setOpacity(self.imageOpacity)

#             lPainter.translate(0.0, float(lPixMap.height()))
            # lPainter.translate(0.0, 100.0)
#             lPainter.scale(1.0, -1.0)

            lPainter.drawPixmap(QtCore.QPoint(0,0), lPixMap)

#             lPainter.translate(0.0, 0.0)
#             lPainter.scale(1.0, 1.0)
#             lPainter.setOpacity(lTmpOpacity)

            # pop the QPainter's saved state off the stack:
            lPainter.restore()

        # the QPainter has to be passed with begin() already called on it:
        # lPainter.begin()
       
        # push the QPainter's current state onto a stack,
        #   to be followed by a restore() below:
        lPainter.save()
       
        lPainter.setRenderHint(QtGui.QPainter.Antialiasing)
       
        # 2010 - Mitja: add freeform shape drawing on the top of image,
        #    where the global for keeping track of what's been changed can be:
        #    0 = Color Pick = CDConstants.ImageModePickColor
        #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
        #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon
        if (self.inputImagePickingMode == CDConstants.ImageModePickColor):
        # this is color-picking mode:
            pass

        elif (self.inputImagePickingMode == CDConstants.ImageModeDrawFreehand):
            # this is freeform drawing mode:
            mouseSpeed = math.sqrt( (self.myMouseXOld-self.myMouseX) * (self.myMouseXOld-self.myMouseX) + \
                                    (self.myMouseYOld-self.myMouseY) * (self.myMouseYOld-self.myMouseY) )
            if (self.myMouseLeftDown == True):
                # add the current point to this image's list for painter path:
                self.theCurrentPath.append( (self.myMouseX, self.myMouseY) )
                lPen = QtGui.QPen()
                if self.myPathHighlight == True:
                    lPen.setColor(QtGui.QColor(QtCore.Qt.red))
                    lPen.setWidth(2)
                else:
                    lPen.setColor(QtGui.QColor(QtCore.Qt.black))
                    lPen.setWidth(0)
                lPen.setCosmetic(True) # cosmetic pen = width always 1 pixel wide, independent of painter's transformation set
                lPainter.setPen(lPen)

                if self.myPathHighlight == True:

                    # to highlight, draw in two colors, solid and dotted:
           
                    lOutlineColor = QtGui.QColor(35, 166, 94)
                    lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                    lPainter.setPen(lOutlinePen)

                    pXa = -1
                    pYa = -1
                    pXb = -1
                    pYb = -1
                    for pX, pY in self.theCurrentPath:
                        if pXa == -1:
                            pXa = pX
                            pYa = pY
                        else:
                            pXb = pXa
                            pYb = pYa
                            pXa = pX
                            pYa = pY
                            lPainter.drawLine(pXb, pYb, pXa, pYa)

                    lOutlineColor = QtGui.QColor(219, 230, 249)
                    lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.DotLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                    lPainter.setPen(lOutlinePen)

                    pXa = -1
                    pYa = -1
                    pXb = -1
                    pYb = -1
                    for pX, pY in self.theCurrentPath:
                        if pXa == -1:
                            pXa = pX
                            pYa = pY
                        else:
                            pXb = pXa
                            pYb = pYa
                            pXa = pX
                            pYa = pY
                            lPainter.drawLine(pXb, pYb, pXa, pYa)
                   
                else:
                    pXa = -1
                    pYa = -1
                    pXb = -1
                    pYb = -1
                    for pX, pY in self.theCurrentPath:
                        if pXa == -1:
                            pXa = pX
                            pYa = pY
                        else:
                            pXb = pXa
                            pYb = pYa
                            pXa = pX
                            pYa = pY
                            lPainter.drawLine(pXb, pYb, pXa, pYa)

                # now we can set the 'old' position vars to current values,
                #   since we just completed a line up to that point:
                self.myMouseXOld = self.myMouseX
                self.myMouseYOld = self.myMouseY

        elif (self.inputImagePickingMode == CDConstants.ImageModeDrawPolygon):
        # this is polygon drawing mode:
            lPen = QtGui.QPen()
            if self.myPathHighlight == True:
                lPen.setColor(QtGui.QColor(QtCore.Qt.red))
                lPen.setWidth(2)
            else:
                lPen.setColor(QtGui.QColor(QtCore.Qt.black))
                lPen.setWidth(2)
            lPen.setCosmetic(True) # cosmetic pen = width always 1 pixel wide, independent of painter's transformation set
            lPainter.setPen(lPen)

            if self.myPathHighlight == True:
                # to highlight, draw in two colors, solid and dotted:
                lOutlineColor = QtGui.QColor(35, 166, 94)
                lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                lPainter.setPen(lOutlinePen)
                if len(self.theCurrentPath) >= 2:
                    pXa = -1
                    pYa = -1
                    pXb = -1
                    pYb = -1
                    for pX, pY in self.theCurrentPath:
                        if pXa == -1:
                            pXa = pX
                            pYa = pY
                        else:
                            pXb = pXa
                            pYb = pYa
                            pXa = pX
                            pYa = pY
                            lPainter.drawLine(pXb, pYb, pXa, pYa)
                lOutlineColor = QtGui.QColor(219, 230, 249)
                lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.DotLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                lPainter.setPen(lOutlinePen)
                if len(self.theCurrentPath) >= 2:
                    pXa = -1
                    pYa = -1
                    pXb = -1
                    pYb = -1
                    for pX, pY in self.theCurrentPath:
                        if pXa == -1:
                            pXa = pX
                            pYa = pY
                        else:
                            pXb = pXa
                            pYb = pYa
                            pXa = pX
                            pYa = pY
                            lPainter.drawLine(pXb, pYb, pXa, pYa)
   
            else:
                # no highlight:
                if len(self.theCurrentPath) >= 2:
                    pXa = -1
                    pYa = -1
                    pXb = -1
                    pYb = -1
                    for pX, pY in self.theCurrentPath:
                        if pXa == -1:
                            pXa = pX
                            pYa = pY
                        else:
                            pXb = pXa
                            pYb = pYa
                            pXa = pX
                            pYa = pY
                            lPainter.drawLine(pXb, pYb, pXa, pYa)
   
            # draw the current (passive) line being moved around until next click:
            if len(self.theCurrentPath) >= 1:
                lPen.setColor(QtGui.QColor(QtCore.Qt.black))
                lPen.setWidth(0)
                lPainter.setPen(lPen)
                lPainter.drawLine(self.myMouseXOld, self.myMouseYOld, self.myMouseX, self.myMouseY)

        self.drawGrid(lPainter)
       
        # pop the QPainter's saved state off the stack:
        lPainter.restore()




    # ------------------------------------------------------------------
    # 2011 - Mitja: this function is NOT to be called directly: it is the callback handler
    #   for update() and paint() events, and it paints into the passed QPainter parameter
    # ------------------------------------------------------------------   
    def paintEvent(self, pThePainter):
   
        # one paint cycle has been called:
        self.repaintEventsCounter = self.repaintEventsCounter + 1

        CDConstants.printOut("[G] hello, I'm "+str(debugWhoIsTheRunningFunction())+", parent is "+str(debugWhoIsTheParentFunction())+ \
            " ||||| self.repaintEventsCounter=="+str(self.repaintEventsCounter)+ \
            " ||||| CDImageLayer.paintEvent(pThePainter=="+str(pThePainter)+")", CDConstants.DebugTODO )

        # 2011 - Mitja: call our function doing the actual drawing,
        #   passing along the QPainter parameter received by paintEvent():
        self.paintTheImageLayer(pThePainter)


    # ------------------------------------------------------------------
    def drawGrid(self, painter):
   
        # here there would be a grid, but we don't want it:
        # draw vertical lines:
#         for x in xrange(0, self.pifSceneWidth, self.fixedRasterWidth):
#             #draw.line([(x, 0), (x, h)], width=2, fill='#000000')
#             painter.setPen(QtGui.QColor(QtCore.Qt.green))
#             painter.drawLine(x, 0, x, self.pifSceneHeight)
#         # draw horizontal lines:
#         for y in xrange(0, self.pifSceneHeight, self.fixedRasterWidth):
#             #draw.line([(0, y), (w, y)], width=2, fill='#000000')
#             painter.setPen(QtGui.QColor(QtCore.Qt.blue))
#             painter.drawLine(0, y, self.pifSceneWidth, y)

        # draw boundary frame lines:

        # this would a solid outline line, we don't want it:
        # painter.setPen(QtGui.QColor(QtCore.Qt.red))

        # draw the rectangular outline in two colors, solid and dotted:

        lOutlineColor = QtGui.QColor(35, 166, 94)
        lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        lOutlinePen.setCosmetic(True) # cosmetic pen = width always 1 pixel wide, independent of painter's transformation set
        painter.setPen(lOutlinePen)

        painter.drawLine(0, 0, 0, self.pifSceneHeight-1)
        painter.drawLine(self.pifSceneWidth-1, 0, self.pifSceneWidth-1, self.pifSceneHeight-1)
        painter.drawLine(0, 0, self.pifSceneWidth-1, 0)
        painter.drawLine(0, self.pifSceneHeight-1, self.pifSceneWidth-1, self.pifSceneHeight-1)

        lOutlineColor = QtGui.QColor(219, 230, 249)
        lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.DotLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        lOutlinePen.setCosmetic(True) # cosmetic pen = width always 1 pixel wide, independent of painter's transformation set
        painter.setPen(lOutlinePen)
        # vertical lines:       
        painter.drawLine(0, 0, 0, self.pifSceneHeight-1)
        painter.drawLine(self.pifSceneWidth-1, 0, self.pifSceneWidth-1, self.pifSceneHeight-1)

        lOutlineColor = QtGui.QColor(255, 0, 0)
        lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.DotLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        lOutlinePen.setCosmetic(True) # cosmetic pen = width always 1 pixel wide, independent of painter's transformation set
        painter.setPen(lOutlinePen)
        # bottom (y=0) line
        painter.drawLine(0, 0, self.pifSceneWidth-1, 0)

        lOutlineColor = QtGui.QColor(0, 255, 255)
        lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.DotLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        lOutlinePen.setCosmetic(True) # cosmetic pen = width always 1 pixel wide, independent of painter's transformation set
        painter.setPen(lOutlinePen)
        # top (y=0) line
        painter.drawLine(0, self.pifSceneHeight-1, self.pifSceneWidth-1, self.pifSceneHeight-1)

#         print "2010 DEBUG:", self.repaintEventsCounter, "CDImageLayer.drawGrid() DONE - WWIIDDTTHH(image,pifScene) =", self.width,self.pifSceneWidth, "HHEEIIGGHHTT(image,pifScene) =", self.height,self.pifSceneHeight
       





    # ---------------------------------------------------------
    # the reject() function is called when the user presses the <ESC> keyboard key.
    #   We override the default and implicitly include the equivalent action of
    #   <esc> being the same as clicking the "Cancel" button, as in well-respected GUI paradigms:
    # ---------------------------------------------------------
    def reject(self):
        super(CDPreferences, self).reject()
        print "reject() DONE"






# FIX ESC KEY in IMAGE DRAWING!!!



    # ------------------------------------------------------------------
    def mousePressEvent(self, pEvent):       
        # 2010 - Mitja: track click events of the mouse left button only:
        if pEvent.button() == QtCore.Qt.LeftButton:

            lX = pEvent.scenePos().x()
            lY = pEvent.scenePos().y()

            # 2010 - Mitja: add freeform shape drawing on the top of image,
            #    where the global for keeping track of what's been changed can be:
            #    0 = Color Pick = CDConstants.ImageModePickColor
            #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
            #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon
            if (self.inputImagePickingMode == CDConstants.ImageModePickColor):
            # this is color-picking mode:
                color = self.theImage.pixel(lX, lY)
                self.myMouseX = lX
                self.myMouseY = lY
                # 2011 - Mitja: to pick a color region in the image for the PIFF scene,
                #   uncomment only ONE of these two methods - either fuzzy pick or close colors:
                self.processImageForFuzzyPick(self.myMouseX, self.myMouseY)
                # self.processImageForCloseColors(self.myMouseX, self.myMouseY)
                self.emit(QtCore.SIGNAL("mousePressedInImageLayerSignal()"))

            elif (self.inputImagePickingMode == CDConstants.ImageModeDrawFreehand):
            # this is freeform drawing mode:

                # 2010 - Mitja: track the position of the mouse pointer at left button press:
                if (self.myMouseLeftDown == False):
                    color = self.theImage.pixel(lX, lY)
                    self.myMouseLeftDown = True
                    self.myMouseX = lX
                    self.myMouseY = lY
                    self.myMouseXOld =  self.myMouseX
                    self.myMouseYOld = self.myMouseY
                    # add the first point to this image's list for painter path:
                    self.theCurrentPath = []
                    self.theCurrentPath.append( (self.myMouseX, self.myMouseY) )

            elif (self.inputImagePickingMode == CDConstants.ImageModeDrawPolygon):
            # this is polygon drawing mode:

                # 2010 - Mitja: track the position of the mouse pointer at left button press:
                if (self.myMouseLeftDown == False):
                    color = self.theImage.pixel(lX, lY)
                    self.myMouseLeftDown = True
                    self.myMouseX = lX
                    self.myMouseY = lY
                    self.myMouseXOld = self.myMouseX
                    self.myMouseYOld = self.myMouseY
                   
                    # add the clicked point to this image's list for a polygonal painter path:
                    self.theCurrentPath.append( (self.myMouseX, self.myMouseY) )
                   
                   

            # 2010 - Mitja: update the CDImageLayer's parent widget,
            #   i.e. paintEvent() will be invoked regardless of the picking mode:
            self.graphicsSceneWindow.scene.update()



            # we pass a "dict" parameter with the signalThatMouseMoved parameter, so that
            #   both the mouse x,y coordinates as well as color information can be passed around:
    
            if ( self.theImage.rect().contains(lX, lY) ):
                lColorAtMousePos = self.theImage.pixel(lX, lY)
            else:
                lColorAtMousePos = QtGui.QColor(255,255,255)
            lDict = { \
                0: str(int(lX)), \
                1: str(int(lY)), \
                #  the depth() function is not part of QGraphicsScene, we add it for completeness:
                2: int( QtGui.QColor(lColorAtMousePos).red() ), \
                3: int( QtGui.QColor(lColorAtMousePos).green() ), \
                4: int( QtGui.QColor(lColorAtMousePos).blue() )
                }
            self.signalThatMouseMoved.emit(lDict)

        # print "                        Color at x: %s\ty: %s is %s" %(self.x, self.y,color)
        # print"2010 DEBUG: CDImageLayer.mousePressEvent() - pEvent(x,y) =",lX, lY

    # end of def mousePressEvent(self, pEvent)
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    def mouseReleaseEvent(self, pEvent):

        lX = pEvent.scenePos().x()
        lY = pEvent.scenePos().y()

        # 2010 - Mitja: add freeform shape drawing on the top of image,
        #    where the global for keeping track of what's been changed can be:
        #    0 = Color Pick = CDConstants.ImageModePickColor
        #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
        #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon
        if (self.inputImagePickingMode == CDConstants.ImageModePickColor):
        # this is color-picking mode:
            pass

        elif (self.inputImagePickingMode == CDConstants.ImageModeDrawFreehand):
        # this is freeform drawing mode:
            # 2010 - Mitja: track release events of the mouse left button only:
            if (pEvent.button() == QtCore.Qt.LeftButton):
                self.myMouseLeftDown = False
                self.myMouseX = lX
                self.myMouseY = lY
   
                # if the current path has at least 3 points,
                #   we can add it to our scene by creating a path and placing it up there. Yay!
                if len(self.theCurrentPath) >= 3:
                    # create a QGraphicsItem from a QPainterPath:
                    thePainterPath = QtGui.QPainterPath()
                    lXa = -1
                    lYa = -1
                    for lXpath, lYpath in self.theCurrentPath:
                        if lXa == -1:
                            lXa = lXpath
                            lYa = lYpath
                            thePainterPath.moveTo(lXpath,lYpath)
                        else:
                            thePainterPath.lineTo(lXpath,lYpath)
                    # now we have a QPainterPath, convert it into a QGraphicsPathItem:
                    theGraphicsPathItem = QtGui.QGraphicsPathItem(thePainterPath)
                    # set its color to have a "magic" RGBA value=0,0,0,0
                    #   which will be recognized by the QGraphicsScene:
                    theGraphicsPathItem.setBrush(QtGui.QColor(0,0,0,0))

                    # pass the QGraphicsItem to the external QGraphicsScene window:
                    self.graphicsSceneWindow.scene.mousePressEvent( \
                        QtGui.QMouseEvent( QtCore.QEvent.GraphicsSceneMousePress, \
                                           QtCore.QPoint(0,0), \
                                           QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier), \
                        theGraphicsPathItem )
                    self.theCurrentPath = []
                    self.myPathHighlight = False
   
                # 2010 - Mitja: update the CDImageLayer's parent widget,
                #   i.e. paintEvent() will be invoked:
                self.graphicsSceneWindow.scene.update()

        elif (self.inputImagePickingMode == CDConstants.ImageModeDrawPolygon):
        # this is polygon drawing mode:
            # 2010 - Mitja: track release events of the mouse left button only:
            if (pEvent.button() == QtCore.Qt.LeftButton):
                self.myMouseLeftDown = False
                self.graphicsSceneWindow.scene.update()

                # if highlighted, add to scene:
                if self.myPathHighlight == True:
                    # create a QGraphicsItem from a QPainterPath:
                    thePainterPath = QtGui.QPainterPath()
                    lXa = -1
                    lYa = -1
                    for lXpath, lYpath in self.theCurrentPath:
                        if lXa == -1:
                            lXa = lXpath
                            lYa = lYpath
                            thePainterPath.moveTo(lXpath,lYpath)
                        else:
                            thePainterPath.lineTo(lXpath,lYpath)
                    # now we have a QPainterPath, convert it into a QGraphicsPathItem:
                    theGraphicsPathItem = QtGui.QGraphicsPathItem(thePainterPath)
                    # set its color to have a "magic" RGBA value=0,0,0,0
                    #   which will be recognized by the QGraphicsScene:
                    theGraphicsPathItem.setBrush(QtGui.QColor(0,0,0,0))

                    # pass the QGraphicsItem to the external QGraphicsScene window:
                    self.graphicsSceneWindow.scene.mousePressEvent( \
                        QtGui.QMouseEvent( QtCore.QEvent.GraphicsSceneMousePress, \
                                           QtCore.QPoint(0,0), \
                                           QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier), \
                        theGraphicsPathItem )
                    self.theCurrentPath = []
                    self.myPathHighlight = False

        # we pass a "dict" parameter with the signalThatMouseMoved parameter, so that
        #   both the mouse x,y coordinates as well as color information can be passed around:

        lColorAtMousePos = QtGui.QColor(255,255,255)
        lDict = { \
            0: str(" "), \
            1: str(" "), \
            #  the depth() function is not part of QGraphicsScene, we add it for completeness:
            2: int( QtGui.QColor(lColorAtMousePos).red() ), \
            3: int( QtGui.QColor(lColorAtMousePos).green() ), \
            4: int( QtGui.QColor(lColorAtMousePos).blue() )
            }
        self.signalThatMouseMoved.emit(lDict)

        print"2012 DEBUG: CDImageLayer.mouseReleaseEvent() - pEvent(x,y) =",lX,lY

    # end of def mouseReleaseEvent(self, pEvent)
    # ---------------------------------------------------------



    # ---------------------------------------------------------
    # 2012 - Mitja - keyReleaseEvent() "fake" event handler
    #   added to handle pressing the <esc> key.
    #   This function is called by DiagramScene's real keyReleaseEvent handler.
    # ---------------------------------------------------------
    def keyReleaseEvent(self, pEvent):

        if (pEvent.key() == QtCore.Qt.Key_Escape):
            #    0 = Color Pick = CDConstants.ImageModePickColor
            #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
            #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon
            if (self.inputImagePickingMode == CDConstants.ImageModeDrawPolygon) or \
               (self.inputImagePickingMode == CDConstants.ImageModeDrawFreehand):
                # reset mouse handling:
                self.myMouseX = -1
                self.myMouseY = -1
                self.myMouseDownX = -1
                self.myMouseDownY = -1
                # "Old" variables are handy to keep previous mouse position value:
                self.myMouseXOld = -1
                self.myMouseYOld = -1
                # you could compute the mouse movement's speed:
                self.myMouseSpeed = 0.0
                # mouse button state, we set it to True if the Left Mouse button is pressed:
                self.myMouseLeftDown = False
                #    highlighting flag, used to provide feedback to the user:
                self.myPathHighlight = False
                # this image's list of vertices for painter path:
                self.theCurrentPath = []

        # 2010 - Mitja: update the CDImageLayer's parent widget,
        #   i.e. paintEvent() will be invoked regardless of the picking mode:
        self.graphicsSceneWindow.scene.update()



    # ---------------------------------------------------------
    def mouseMoveEvent(self, pEvent):
    
        # 2010 - Mitja: add freeform shape drawing on the top of image,
        #    where the global for keeping track of what's been changed can be:
        #    0 = Color Pick = CDConstants.ImageModePickColor
        #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
        #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon

        lX = pEvent.scenePos().x()
        lY = pEvent.scenePos().y()

        if (self.inputImagePickingMode == CDConstants.ImageModePickColor):
        # this is color-picking mode:
            if (self.myMouseLeftDown == False) and \
              (self.graphicsSceneWindow.view.hasMouseTracking() == True):
                # print lX, lY
                # 2011 - Mitja: to have real-time visual feedback while moving the mouse,
                #   without actually picking a color region in the image for the PIFF scene,
                #   uncomment one of these two methods:
                # self.processImageForFuzzyPick(lX, lY)
                # self.processImageForCloseColors(lX, lY)
                # self.graphicsSceneWindow.scene.update()
                pass
        elif (self.inputImagePickingMode == CDConstants.ImageModeDrawFreehand):
        # this is freeform drawing mode:
            if (self.myMouseLeftDown == True) and \
              (self.graphicsSceneWindow.view.hasMouseTracking() == False):
                # 2010 - Mitja: track mouse move events only when the mouse left button is depressed:
                self.myMouseX = lX
                self.myMouseY = lY
                # check for closeness to 1st point in the drawing:
                lTmpX, lTmpY = self.theCurrentPath[0]
                if (abs(self.myMouseX - lTmpX) < 3) and \
                   (abs(self.myMouseY - lTmpY) < 3):
                    self.myPathHighlight = True
                else:
                    self.myPathHighlight = False
                # 2010 - Mitja: update the CDImageLayer's parent widget,
                #   i.e. paintEvent() will be invoked:
                self.graphicsSceneWindow.scene.update()
        elif (self.inputImagePickingMode == CDConstants.ImageModeDrawPolygon):
        # this is polygon drawing mode:
            if (self.myMouseLeftDown == False) and \
              (self.graphicsSceneWindow.view.hasMouseTracking() == True):
                # 2010 - Mitja: track mouse move events only when the mouse left button is NOT depressed:
                self.myMouseX = lX
                self.myMouseY = lY
                # print self.myMouseX, self.myMouseY
                # check for closeness to 1st point in the drawing,
                #   if the current path has at least 3 points:
                if (len(self.theCurrentPath) >= 3):
                    lTmpX, lTmpY = self.theCurrentPath[0]
                    if (abs(self.myMouseX - lTmpX) < 3) and \
                        (abs(self.myMouseY - lTmpY) < 3):
                        self.myPathHighlight = True
                    else:
                        self.myPathHighlight = False
                # 2010 - Mitja: update the CDImageLayer's parent widget,
                #   i.e. paintEvent() will be invoked:
                self.graphicsSceneWindow.scene.update()




        # we pass a "dict" parameter with the signalThatMouseMoved parameter, so that
        #   both the mouse x,y coordinates as well as color information can be passed around:

        if ( self.theImage.rect().contains(lX, lY) ):
            lColorAtMousePos = self.theImage.pixel(lX, lY)
        else:
            lColorAtMousePos = QtGui.QColor(255,255,255)
        lDict = { \
            0: str(int(lX)), \
            1: str(int(lY)), \
            #  the depth() function is not part of QGraphicsScene, we add it for completeness:
            2: int( QtGui.QColor(lColorAtMousePos).red() ), \
            3: int( QtGui.QColor(lColorAtMousePos).green() ), \
            4: int( QtGui.QColor(lColorAtMousePos).blue() )
            }
        self.signalThatMouseMoved.emit(lDict)


if __name__ == '__main__':

    import sys

    app = QtGui.QApplication(sys.argv)
    window = CDImageLayer()
    window.show()
    window.raise_()
    sys.exit(app.exec_())


# Local Variables:
# coding: US-ASCII
# End:
