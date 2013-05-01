#!/usr/bin/env python

from PyQt4 import QtCore, QtGui


from PyQt4 import QtGui # from PyQt4.QtGui import *
from PyQt4 import QtCore # from PyQt4.QtCore import *

import math   # we need to import math to use sqrt() and such
import sys    # for get/setrecursionlimit() and sys.version_info

# 2012 - Mitja: advanced debugging tools,
#   work only on Posix-compliant systems so far (no MS-Windows)
#   commented out for now:
# import os     # for kill() and getpid()
# import signal # for signal()

from timeit import Timer    #    for timing testing purposes

import time    # for sleep()

# import random  # for semi-random-colors

import numpy   # to have arrays!


# 2010 - Mitja:
import inspect # for debugging functions, remove in final version
# debugging functions, remove in final version
def debugWhoIsTheRunningFunction():
    return inspect.stack()[1][3]
def debugWhoIsTheParentFunction():
    return inspect.stack()[2][3]




# 2012 - Mitja: for print_stack()
import traceback

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants



# ======================================================================
# 2012- Mitja: image processing in a NumPy array
# ======================================================================
class CDImageNP(QtCore.QObject):

    # 2011 - Mitja: a signal for image data resizing. Has to be handled in CDDiagramScene!
    signalThatImageNPResized = QtCore.pyqtSignal(dict)

    # 2011 - Mitja: a signal for the setting of self.__theCurrentIndex. Has to be handled in CDDiagramScene!
    signalThatCurrentIndexSet = QtCore.pyqtSignal(dict)


    # --------------------------------------------------------
    def __init__(self, pParent=None):

        CDConstants.printOut( "___ - DEBUG ----- CDImageNP.__init__( pParent == "+str(pParent)+") starting." , CDConstants.DebugExcessive )

        QtCore.QObject.__init__(self, pParent)

        # a flag to show if CDImageNP holds an image loaded from files:
        self.__imageNPLoaded = False

        # the class global keeping track of the selected image within the image processing:
        #    0 = minimum = the first image in the sequence stack
        self.__theCurrentIndex = 0
        
        # the class global keeping track of the color picked from the image and used by image processing:
        self.__thePickedNPColor = QtGui.QColor(QtCore.Qt.transparent)

        # the class global keeping track of the selected color/type for regions coming from image processing:
        self.__theImageNPColor = QtGui.QColor(QtCore.Qt.magenta)

        # the class global keeping track of the "wall" color/type for  for regions coming from image processing:
        self.__theImageNPWallColor = QtGui.QColor(QtCore.Qt.green)


        # the class global keeping track of the bit-flag modes for extracting cell areas from the displayed imported image:
        #    0 = Use Discretized Images to BW = CDConstants.ImageNPUseDiscretizedToBWMode
        #    1 = Extract Cell from Single-Color Areas = CDConstants.ImageNPExtractSingleColorCells
        self.__theProcessingModeForImageNP = (1 << CDConstants.ImageNPExtractSingleColorCells)

        # bin() does not exist in Python 2.5:
        if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
            CDConstants.printOut( "___ - DEBUG ----- CDImageNP.__init__() - bin(self.__theProcessingModeForImageNP) == "+str(bin(self.__theProcessingModeForImageNP)) , CDConstants.DebugExcessive )
        else:
            CDConstants.printOut( "___ - DEBUG ----- CDImageNP.__init__() - self.__theProcessingModeForImageNP == "+str(self.__theProcessingModeForImageNP) , CDConstants.DebugExcessive )



        # the size of all images (width=x, height=y) in the image processing
        #   and number of images (z) loaded in a stack/array
        #   and number of color channels (i.e. image depth, in bytes)
        #     in the *original* image loaded from file:
        self.__sizeX = 0
        self.__sizeY = 0
        self.__sizeZ = 0
        self.__cChannels = 0

        # 2011 - Mitja: globals storing pixel-size data for all image processing:
        #
        # numpy-based array globals of size 1x1x1x1,
        #     will be resized when the image is loaded, i.e.
        #     the actually used arrays will be set in resetNPDimensions()
        # the 4 dimensions are, from the slowest (most distant)
        #     to the fastest (closest data to each other):
        #   z = image layers, y = height, x = width, [b g r], even if image may be [a b g r]
        self.imageNPArray = numpy.zeros( (1, 1, 1, 1), dtype=numpy.uint8 )
        self.discretizedNPArray = numpy.zeros( (1, 1, 1, 1), dtype=numpy.uint8 )
        self.extractedCellDataNPArray = numpy.zeros( (1, 1, 1, 1), dtype=numpy.uint8 )

        # another couple of numpy arrays to store flags (data valid=>True, data invalid=>False)
        #    where each boolean flag in these arrays corresponds to one image/layer
        #    in the original image array and one image/layer in the extracted cell data array
        self.imageInNPIsReadyFlags = numpy.zeros( (1), dtype=numpy.bool )
        self.extractedCellDataNPIsReadyFlags = numpy.zeros( (1), dtype=numpy.bool )

        # and a dict to store all filename strings, one per image layer
        self.imageNPFileNames = dict()
        self.imageNPFileNames[0] = " "

        # just reset image-numpy-processing-related-globals to some boring initial defaults:
        lTmpWidth = 120
        lTmpHeight = 90
        lTmpDepth = 1
        lTmpImageChannels = 4
        self.resetNPDimensions( lTmpWidth, lTmpHeight, lTmpDepth, lTmpImageChannels )

        # the "theCurrentImage" QImage is the selected image loaded from a file:
        lBoringPixMap = QtGui.QPixmap(lTmpWidth, lTmpHeight)
        lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.transparent) )
        self.theCurrentImage = QtGui.QImage(lBoringPixMap)

        # the "theCurrentDiscretizedImage" QImage is the selected discretized image as obtained from the file image loaded from a file:
        lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.transparent) )
        self.theCurrentDiscretizedImage = QtGui.QImage(lBoringPixMap)

        # the "theCurrentExtractedCellDataImage" QImage is the selected extracted cell-data-image detected from the file image:
        lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.transparent) )
        self.theCurrentExtractedCellDataImage = QtGui.QImage(lBoringPixMap)

        if isinstance( pParent, QtGui.QWidget ) == True:
            self._graphicsSceneWidget = pParent
        else:
            self._graphicsSceneWidget = None

        # the progress bar widget is instantiated in the CellDrawMainWindow class,
        #   and assigned below in setSimpleProgressBarPanel() :
        self.__theSimpleWaitProgressBar = None

        # the progress bar with image widget is instantiated in the CellDrawMainWindow class,
        #   and assigned below in setProgressBarWithImagePanel() :
        self.__theWaitProgressBarWithImage = None

        CDConstants.printOut( "___ - DEBUG ----- CDImageNP.__init__() - _graphicsSceneWidget == "+str(self._graphicsSceneWidget)+" " , CDConstants.DebugExcessive )
        CDConstants.printOut( "___ - DEBUG ----- CDImageNP.__init__() - imageInNPIsReadyFlags == "+str(self.imageInNPIsReadyFlags)+" " , CDConstants.DebugExcessive )
        CDConstants.printOut( "___ - DEBUG ----- CDImageNP.__init__() - extractedCellDataNPIsReadyFlags == "+str(self.extractedCellDataNPIsReadyFlags)+" " , CDConstants.DebugExcessive )

        # debugging counter, how many times has the paint function been called:
        self.repaintEventsCounter = 0

        CDConstants.printOut( "    - DEBUG ----- CDImageNP.__init__(): done.", CDConstants.DebugExcessive )

    # end of  def __init__(self, pParent=None):
    # --------------------------------------------------------





    # --------------------------------------------------------
    def setSimpleProgressBarPanel(self, pSimpleProcessBar=None):
    # --------------------------------------------------------
        if isinstance( pSimpleProcessBar, QtGui.QWidget ) == True:
            self.__theSimpleWaitProgressBar = pSimpleProcessBar
        else:
            self.__theSimpleWaitProgressBar = None
    # end of   def setSimpleProgressBarPanel()
    # --------------------------------------------------------



    # --------------------------------------------------------
    def setProgressBarWithImagePanel(self, pProcessBarWithImage=None):
    # --------------------------------------------------------
        if isinstance( pProcessBarWithImage, QtGui.QWidget ) == True:
            self.__theWaitProgressBarWithImage = pProcessBarWithImage
        else:
            self.__theWaitProgressBarWithImage = None
    # end of   def setProgressBarWithImagePanel()
    # --------------------------------------------------------




    # --------------------------------------------------------
    def setNPImage(self, pImage=None, pIsImageLoadedFromFile=False, pImageFileName=" "):

        CDConstants.printOut( "___ - DEBUG ----- CDImageNP.setNPImage( pImage == "+str(pImage)+"  ) starting. " , CDConstants.DebugExcessive )
        QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)

        # the "lUserSelectedImage" QImage is the image loaded from a file:
        if isinstance(pImage, QtGui.QImage):
            lUserSelectedImage = QtGui.QImage(pImage)
        else: 
            CDConstants.printOut( "___ - _____ ----- CDImageNP.setNPImage( pImage == "+str(pImage)+" is not a valid QImage ) ...doing nothing. " , CDConstants.DebugImportant )
            return

        # temporarily disable drawing the scene overlay:
        self._graphicsSceneWidget.scene.setDrawForegroundEnabled(False)

        # set image-related globals:
        self.__imageNPLoaded = pIsImageLoadedFromFile
        self.setCurrentFileName(pImageFileName)

        lXdim = int(lUserSelectedImage.width())
        lYdim = int(lUserSelectedImage.height())
        lZdim = int(1)
        if lUserSelectedImage.format() in (QtGui.QImage.Format_ARGB32_Premultiplied, QtGui.QImage.Format_ARGB32):
            lImageChannels = 4
        elif (lUserSelectedImage.format() == QtGui.QImage.Format_RGB32):
            lImageChannels = 3
        else: 
            CDConstants.printOut( "___ - _____ ----- CDImageNP.setNPImage( pImage == "+str(pImage)+" ) is in unsupported Qt format: " +str(lUserSelectedImage.format())+ "...doing nothing. " , CDConstants.DebugImportant )
            return
            
        # update x,y,z dimensions in the CDImageNP object,
        #    this will also reset all the CDImageNP object's numpy array,
        #    and all the entries in the edge/imageInNPIsReadyFlags arrays:
        self.resetNPDimensions(lXdim, lYdim, lZdim, lImageChannels)

        # tell the CDImageNP object to extract single-color cells from the image loaded from file:
        self.resetToOneProcessingModeForImageNP( CDConstants.ImageNPExtractSingleColorCells )

        if (self.__imageNPLoaded == True):



          # def debug_trace():
            '''Set a tracepoint in the Python debugger that works with Qt'''
            from PyQt4.QtCore import pyqtRemoveInputHook
            from pdb import set_trace
            pyqtRemoveInputHook()
            set_trace()

            # ## ## ## CDConstants.printOut(  "CDImageNP.setNPImage()  --  1.", CDConstants.DebugTODO )
            QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)
            #   time.sleep(3.0)

            # show a panel containing a progress bar:        
            self.__theWaitProgressBarWithImage.setTitleTextRange("Processing image.", " ", 0, self.__sizeY)
            self.__theWaitProgressBarWithImage.setInfoText("Processing image...")
            self.__theWaitProgressBarWithImage.show()
            lFileCounter = 0

            # if there is an image loaded from a file, show a pixmap in the progress bar dialog window:
            llPixmapForProgressBarPanelImageNP = QtGui.QPixmap( lUserSelectedImage )

            # ## ## ## CDConstants.printOut(  "CDImageNP.setNPImage()  --  2.", CDConstants.DebugTODO )
            QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)
            #   time.sleep(3.0)

            pass # 154 prrint    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"
#             print    "DOPO DE QUA CAMBIA"


            pass # 154 prrint    "DORMO 0"
            #   time.sleep(3.0)
            self.__theWaitProgressBarWithImage.setImagePixmap(llPixmapForProgressBarPanelImageNP)
            pass # 154 prrint    "DORMO 1"
            #   time.sleep(3.0)
            pass # 154 prrint "CDImageNP.setNPImage() - self.__sizeX =",self.__sizeX
            pass # 154 prrint    "DORMO 2"
            #   time.sleep(3.0)
            pass # 154 prrint "CDImageNP.setNPImage() - self.__sizeY =",self.__sizeY
            pass # 154 prrint    "DORMO 3"
            #   time.sleep(3.0)
            pass # 154 prrint "CDImageNP.setNPImage() - self.__sizeZ =",self.__sizeZ    
            pass # 154 prrint    "DORMO 4"
            #   time.sleep(3.0)
    
            # ## ## ## CDConstants.printOut(  "CDImageNP.setNPImage()  --  3.", CDConstants.DebugTODO )
            pass # 154 prrint    "DORMO 5"
            #   time.sleep(3.0)
            QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)
            pass # 154 prrint    "DORMO 6"
            #   time.sleep(3.0)
            pass # 154 prrint    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
#             print    "PRIMA DE QUA ZA CAMBIA"
            self.__theWaitProgressBarWithImage.setValue(lFileCounter)
            #   time.sleep(3.0)
    
            # set the current index in the imageNPArray and other NP arrays:
            self.setCurrentIndexWithoutUpdatingGUI(lFileCounter)

            # assign the image loaded from a file to the current image global, and to the image values in the imageNPArray:
            self.setCurrentImageAndNPArrayLayer(lUserSelectedImage)
    
            lFileCounter = lFileCounter + 1
    
    
            # ## ## ## CDConstants.printOut(  "CDImageNP.setNPImage()  --  4.", CDConstants.DebugTODO )
            #   time.sleep(3.0)
    
            # hide the panel containing a progress bar:
            self.__theWaitProgressBarWithImage.maxProgressBar()
            self.__theWaitProgressBarWithImage.hide()

            # ## ## ## CDConstants.printOut(  "CDImageNP.setNPImage()  --  5.", CDConstants.DebugTODO )
            #   time.sleep(3.0)
        # end of   if (self.__imageNPLoaded == True).


        # now confirm that the theCDImageNP contains data loaded from files:
        # NO! this could be a default image for example:
        # self.setNPLoadedFromFiles(True)


        # 2012 - do NOT update the regionUseDict since it contains the list of all
        #  region colors in use by our scene - and we're only setting the image for processing here,
        #  not extracting any regions yet:
        # self._graphicsSceneWidget.scene.addToRegionColorsInUse( self.getNPCurrentColor()  )


###### fix the paintEvent() vs. update() or repaint() for ALLL calls to paintEvent!!!
# see...
#   http://doc.qt.nokia.com/latest/qwidget.html#paintEvent
#

#         1. set a proper example name for the sequence's content (color) type name in the DICT table:
#             get table dict
#             update type name from color
#             return table dict

        # 2012 - do NOT update the regionUseDict since it contains the list of all
        #  region colors in use by our scene - and we're only setting the image for processing here,
        #  not extracting any regions yet:
        # self._graphicsSceneWidget.scene.addToRegionColorsInUse( self.getNPWallColor()  )



#         2. set a proper example name for the sequence's wall (color) type name in the DICT table:
#             same as for content
        
#         3. fix sequence display so that it uses the chosen content color for rendering of 2D images
#             probably after normalizing

#         self.normalizeAllImages()


        # warn user about image dimensions being different than PIF scene dimensions,
        #   ask what to do next:
        if  (self.__imageNPLoaded == True) and \
             ( (self._graphicsSceneWidget.cdPreferences.getPifSceneWidth() != self.__sizeX) or \
               (self._graphicsSceneWidget.cdPreferences.getPifSceneHeight() != self.__sizeY) or \
               (self._graphicsSceneWidget.cdPreferences.getPifSceneDepth() != self.__sizeZ)        ):
            pass # 154 prrint "CDImageNP.setNPImage() - self.__imageNPLoaded = ", self.__imageNPLoaded

            lNewSceneMessageBox = QtGui.QMessageBox(self._graphicsSceneWidget)
            lNewSceneMessageBox.setWindowModality(QtCore.Qt.WindowModal)
            lNewSceneMessageBox.setIcon(QtGui.QMessageBox.Warning)
            # the "setText" sets the main large print text in the dialog box:
            lTheText = "Current cell scene dimensions differ\n from imported image dimensions.\n" + \
                "Resize scene?"
            lNewSceneMessageBox.setText(lTheText)
            # the "setInformativeText" sets a smaller print text, below the main large print text in the dialog box:
            lTheText = " Cell Scene = [ " + str(self._graphicsSceneWidget.cdPreferences.getPifSceneWidth())+ " , " \
                " " + str(self._graphicsSceneWidget.cdPreferences.getPifSceneHeight())+ " , " \
                " " + str(self._graphicsSceneWidget.cdPreferences.getPifSceneDepth())+ " ] \n" \
                " Imported Image = [ " + str(self.__sizeX)+ " , " \
                " " + str(self.__sizeY)+ " , " \
                " " + str(self.__sizeZ)+ " ] \n"
            lNewSceneMessageBox.setInformativeText(lTheText)
            lTheResizeButton = lNewSceneMessageBox.addButton( "Resize Scene" ,  QtGui.QMessageBox.ActionRole)
            lTheCancelButton = lNewSceneMessageBox.addButton( "Cancel" ,  QtGui.QMessageBox.RejectRole)
            lNewSceneMessageBox.setDefaultButton( lTheResizeButton )
            lNewSceneMessageBox.exec_()
            lUserChoice = lNewSceneMessageBox.clickedButton()
    
            if lUserChoice == lTheResizeButton:
                self._graphicsSceneWidget.cdPreferences.setPifSceneWidth(self.__sizeX)
                self._graphicsSceneWidget.cdPreferences.setPifSceneHeight(self.__sizeY)
                self._graphicsSceneWidget.cdPreferences.setPifSceneDepth(self.__sizeZ)


        # finally set the current index to 0 i.e. the initial image in the sequence:
        # self.setCurrentIndexInImageNP(0)
        self.setCurrentIndexWithoutUpdatingGUI(0)

        # re-enable drawing the scene overlay:
        self._graphicsSceneWidget.scene.setDrawForegroundEnabled(True)

        CDConstants.printOut( "    - DEBUG ----- CDImageNP.setNPImage(): done.", CDConstants.DebugExcessive )

    # end of  def setNPImage(self, pImage=None)
    # --------------------------------------------------------



    # ------------------------------------------------------------------
    # 2011 - Mitja: setCurrentFileName() justs sets the filename string into the
    #     "self.__theCurrentIndex" key entry in the  self.imageNPFileNames  dict.
    # ------------------------------------------------------------------   
    def setCurrentFileName(self, pName):
        self.imageNPFileNames[self.__theCurrentIndex] = pName


    # ------------------------------------------------------------------
    # 2011 - Mitja: imageCurrentImageNP() creates theCurrentImage, theCurrentExtractedCellDataImage and theCurrentDiscretizedImage
    #   from the current layer in the sequence arrays:
    # ------------------------------------------------------------------   
    def imageCurrentImageNP(self):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.imageCurrentImageNP(): starting.", CDConstants.DebugExcessive )

        # do nothing if the current array size isn't anything:
        if (self.__sizeX <= 0) or (self.__sizeY <= 0) or (self.__sizeZ <= 0) :
            return

        # do nothing if the current image index doesn't have a corresponding image:
        if (self.__theCurrentIndex < 0) or (self.__theCurrentIndex >= self.__sizeZ) :
            return

        # obtain the current image data from one layer in the image numpy array 
        lTmpOneLayerArray = self.imageNPArray[self.__theCurrentIndex]
        self.theCurrentImage = self.rgb2qimage(lTmpOneLayerArray)

        # obtain the current volume slice data from one layer in the volume numpy array 
        lTmpOneLayerArray = self.discretizedNPArray[self.__theCurrentIndex]
        self.theCurrentDiscretizedImage = self.rgb2qimageKtoBandA(lTmpOneLayerArray)

        # check if the current image has its edge already computed; if it doesn't, then compute it now:
        if (self.extractedCellDataNPIsReadyFlags[self.__theCurrentIndex] == False):
            # self.__theTrueComputeCurrentEdge()
            # TODO: calling computeCurrentEdge() is to be used just for testing __theTrueComputeCurrentEdge() with a timer, afterwards revert to calling __theTrueComputeCurrentEdge() directly:
            self.computeCurrentEdge()
        # obtain the current edge data from one layer in the edge numpy array 
        lTmpOneLayerArray = self.extractedCellDataNPArray[self.__theCurrentIndex]
        self.theCurrentExtractedCellDataImage = self.rgb2qimageWtoRandA(lTmpOneLayerArray)


        CDConstants.printOut( "___ - DEBUG ----- CDImageNP.imageCurrentImageNP() self.__theCurrentIndex == "+str(self.__theCurrentIndex)+ " DONE." , CDConstants.DebugVerbose )
    # end of  def imageCurrentImageNP(self)
    # ------------------------------------------------------------------   




    # ------------------------------------------------------------------   
    # 2012 - Mitja: paintTheImageNPContent() paints/draws all that has been computed by
    #   the Image Sequence, NumPy image processing routines, if computed,
    #   and may be called directly or by our paintEvent() handler:
    def paintTheImageNPContent(self, pThePainter):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.paintTheImageNPContent( pThePainter=="+str(pThePainter)+" ): starting.", CDConstants.DebugExcessive )

        # one paint cycle has been called:
        self.repaintEventsCounter = self.repaintEventsCounter + 1

        # paint into the passed QPainter parameter:
        lPainter = pThePainter
        # the QPainter has to be passed with begin() already called on it:
        # lPainter.begin()

        # push the QPainter's current state onto a stack, to be followed by a restore() below:
        lPainter.save()

        # draw image's full area regions, their computed edges, etc. according to the users' choosen GUI buttons:
        #
        if ( self.getAProcessingModeStatusForImageNP(CDConstants.ImageNPExtractSingleColorCells) ) :

            # draw the selected image, if there is one:
            if isinstance( self.theCurrentImage, QtGui.QImage ) == True:
#               lPixMap = QtGui.QPixmap.fromImage(self.theCurrentImage)
                lPainter.drawPixmap(QtCore.QPoint(0,0), lPixMap)
                CDConstants.printOut(  "paintTheImageNPContent()  --  --  CDConstants.ImageSequenceUseAreaSeeds TRUE, painted:  self.theCurrentImage", CDConstants.DebugTODO )
            pass



        # pop the QPainter's saved state off the stack:
        lPainter.restore()

        CDConstants.printOut( "    - DEBUG ----- CDImageNP.paintTheImageNPContent(): done.", CDConstants.DebugExcessive )
    # end of    def paintTheImageNPContent(self, pThePainter)
    # ------------------------------------------------------------------   




    # ------------------------------------------------------------------
    # 2011 - Mitja: imageNPpreview() computes a layer based on a color:
    # ------------------------------------------------------------------   
    def imageNPpreview(self, pX, pY, pR, pG, pB):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.imageNPpreview(): starting.", CDConstants.DebugExcessive )

        # obtain the current image data from one layer in the image numpy array 
        lTmpOneLayerArray = self.imageNPArray[self.__theCurrentIndex]

        h, w, channels = lTmpOneLayerArray.shape
    
        # Qt expects 32bit BGRA data for color images:
        lRarray = numpy.empty((h, w), numpy.uint8, 'C')
        lGarray = numpy.empty((h, w), numpy.uint8, 'C')
        lBarray = numpy.empty((h, w), numpy.uint8, 'C')

        lRflags = numpy.zeros((h, w), numpy.uint8, 'C')
        lGflags = numpy.zeros((h, w), numpy.uint8, 'C')
        lBflags = numpy.zeros((h, w), numpy.uint8, 'C')
        lAflags = numpy.zeros((h, w), numpy.uint8, 'C')

        lBarray = lTmpOneLayerArray[...,0]
        lGarray = lTmpOneLayerArray[...,1]
        lRarray = lTmpOneLayerArray[...,2]

        lRflags[lRarray==pR] = 1
        lGflags[lGarray==pG] = 1
        lBflags[lBarray==pB] = 1

# ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()   =====>

        lAflags =  numpy.select( [(lRarray==pR) and (lGarray==pG) and (lBarray==pB), True], \
                                 [255, 0], 0)

        # Qt expects 32bit BGRA data for color images:
        bgra = numpy.empty((h, w, 4), numpy.uint8, 'C')
        bgra[...,0] = lBflags
        bgra[...,1] = lGflags
        bgra[...,2] = lRflags
        bgra[...,3] = lAflags

        fmt = QtGui.QImage.Format_ARGB32

        lResultImage = QtGui.QImage(bgra.data, w, h, fmt)
        lResultImage.ndarray = bgra

        lTmpPixmap = QtGui.QPixmap.fromImage(lResultImage)
        lTmpArrayBGRA = self.qimage2numpy(QtGui.QImage(lTmpPixmap.toImage()), "array")

        self.extractedCellDataNPArray[self.__theCurrentIndex] = lTmpArrayBGRA
        self.extractedCellDataNPIsReadyFlags[self.__theCurrentIndex] = True

        CDConstants.printOut( "    - DEBUG ----- CDImageNP.imageNPpreview(): done.", CDConstants.DebugExcessive )

    # end of def imageNPpreview(self)
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # 2011 - Mitja: computeCurrentEdge() computes edge detection on the current image:
    # ------------------------------------------------------------------   
    def computeCurrentEdge(self):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.computeCurrentEdge(): starting.", CDConstants.DebugExcessive )

        # only really compute the current edge if it hasn't been computed yet:
        if (self.extractedCellDataNPIsReadyFlags[self.__theCurrentIndex] == False):
            
            # adjusting Timer() setup for Python 2.5:
            if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
                timerMeasureForFunction = Timer(self.__theTrueComputeCurrentEdge)  # define it before the try/except
                try:
                    lTheTimeItTook = timerMeasureForFunction.timeit(1)           # or timerMeasureForFunction.repeat(...)
                    CDConstants.printOut( str(lTheTimeItTook)+ \
                        " seconds it took __theTrueComputeCurrentEdge(), self.__theCurrentIndex == " + \
                        str(self.__theCurrentIndex) + " in CDImageNP.computeCurrentEdge()" , CDConstants.DebugVerbose )
                except:
                    CDConstants.printOut( "CDImageNP.computeCurrentEdge() code exception!   self.__theCurrentIndex == "+str(self.__theCurrentIndex) , CDConstants.DebugSparse )
                    timerMeasureForFunction.print_exc()
                    CDConstants.printOut( "CDImageNP.computeCurrentEdge() code exception!   self.__theCurrentIndex == "+str(self.__theCurrentIndex) , CDConstants.DebugSparse )
            else:
                # timerMeasureForFunction = Timer('__theTrueComputeCurrentEdge()', 'from CDImageNP import __theTrueComputeCurrentEdge')  # define it before the try/except
                self.__theTrueComputeCurrentEdge()

        CDConstants.printOut( "    - DEBUG ----- CDImageNP.computeCurrentEdge(): done.", CDConstants.DebugExcessive )

    # end of def computeCurrentEdge(self)
    # ------------------------------------------------------------------






    # ------------------------------------------------------------------
    # Uses hashes of tuples to simulate 2-d arrays for the masks. 
    # ------------------------------------------------------------------
    def get_prewitt_masks(self): 
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.get_prewitt_masks(): starting.", CDConstants.DebugExcessive )
        xmask = {} 
        ymask = {} 
     
        xmask[(0,0)] = -1 
        xmask[(0,1)] = 0 
        xmask[(0,2)] = 1 
        xmask[(1,0)] = -1 
        xmask[(1,1)] = 0 
        xmask[(1,2)] = 1 
        xmask[(2,0)] = -1 
        xmask[(2,1)] = 0 
        xmask[(2,2)] = 1 
     
        ymask[(0,0)] = 1 
        ymask[(0,1)] = 1 
        ymask[(0,2)] = 1 
        ymask[(1,0)] = 0 
        ymask[(1,1)] = 0 
        ymask[(1,2)] = 0 
        ymask[(2,0)] = -1 
        ymask[(2,1)] = -1 
        ymask[(2,2)] = -1 
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.get_prewitt_masks(): returning xmask="+str(xmask)+ " ymask="+str(ymask)+" DONE.", CDConstants.DebugVerbose )
        return (xmask, ymask) 




    # ------------------------------------------------------------------
    # http://en.wikipedia.org/wiki/Prewitt_operator
    # ------------------------------------------------------------------
    #     Now on to the meat of the entire operation. The prewitt() function
    #     takes a 1-d array of pixels and the width and height of the input
    #     image. It returns a greyscale edge map image.
    # 
    #     You can store this code all in one file so when you run it, you can
    #     pass the program arguments for the input and output image filenames
    #     on the command line. To do so, add this code to the Python file with
    #     the edge detection code from earlier:
    #    
    #     import sys 
    #     if __name__ == '__main__': 
    #         img = Image.open(sys.argv[1]) 
    #         # only operates on greyscale images 
    #         if img.mode != 'L': img = img.convert('L') 
    #         pixels = list(img.getdata()) 
    #         w, h = img.size 
    #         outimg = prewitt(pixels, w, h) 
    #         outimg.save(sys.argv[2]) 
    #
    #     import Image 
     
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def __theTrueComputeCurrentEdge(self): 
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.__theTrueComputeCurrentEdge(): starting.", CDConstants.DebugExcessive )

        if ( self.getAProcessingModeStatusForImageNP(CDConstants.ImageNPUseDiscretizedToBWMode) == True ) :
            # if discretizing to black/white, the "volume array" values are to be used:

            # show a panel containing a progress bar:        
            self.__theSimpleWaitProgressBar.setTitleTextRange("cdImageLayer: Computing edge detection on volume.", " ", 0, self.theCurrentDiscretizedImage.height())
            self.__theSimpleWaitProgressBar.show()
    
            # set the flag for the current edge array in the sequence as False, as we're computing it now:
            self.extractedCellDataNPIsReadyFlags[self.__theCurrentIndex] = False
    
            xmask, ymask = self.get_prewitt_masks() 
    
            width = self.theCurrentDiscretizedImage.width()
            height = self.theCurrentDiscretizedImage.height()
    
            # create a new greyscale image for the output 
    #         outimg = Image.new('L', (width, height)) 
    #         outpixels = list(outimg.getdata()) 
         
            for y in xrange(height): 
    
                self.__theSimpleWaitProgressBar.setValue(y)
    
                for x in xrange(width): 
                    sumX, sumY, magnitude = 0, 0, 0 
         
                    if y == 0 or y == height-1: magnitude = 0 
                    elif x == 0 or x == width-1: magnitude = 0 
                    else: 
                        for i in xrange(-1, 2): 
                            for j in xrange(-1, 2): 
                                lX = x+i
                                lY = y+j
                                # convolve the image pixels with the Prewitt mask,
                                #     approximating dI/dx 
                                r = int(self.discretizedNPArray[self.__theCurrentIndex, lY, lX, 0])
                                g = int(self.discretizedNPArray[self.__theCurrentIndex, lY, lX, 1])
                                b = int(self.discretizedNPArray[self.__theCurrentIndex, lY, lX, 2])
                                gray = (r + g + b) / 3
                                sumX += (gray) * xmask[i+1, j+1] 
         
                        for i in xrange(-1, 2): 
                            for j in xrange(-1, 2): 
                                lX = x+i
                                lY = y+j
                                # convolve the image pixels with the Prewitt mask,
                                #     approximating dI/dy 
                                r = int(self.discretizedNPArray[self.__theCurrentIndex, lY, lX, 0])
                                g = int(self.discretizedNPArray[self.__theCurrentIndex, lY, lX, 1])
                                b = int(self.discretizedNPArray[self.__theCurrentIndex, lY, lX, 2])
                                gray = (r + g + b) / 3
                                sumY += (gray) * ymask[i+1, j+1] 
         
                    # approximate the magnitude of the gradient 
                    magnitude = abs(sumX) + abs(sumY)
         
                    if magnitude > 255: magnitude = 255 
                    if magnitude < 0: magnitude = 0 
    
                    self.extractedCellDataNPArray[self.__theCurrentIndex, y, x, 0] = numpy.uint8 (255 - magnitude)
                    self.extractedCellDataNPArray[self.__theCurrentIndex, y, x, 1] = numpy.uint8 (255 - magnitude)
                    self.extractedCellDataNPArray[self.__theCurrentIndex, y, x, 2] = numpy.uint8 (255 - magnitude)
    
            # set the flag for the current edge array in the sequence as True, as we're computing it now:
            self.extractedCellDataNPIsReadyFlags[self.__theCurrentIndex] = True
    
            self.__theSimpleWaitProgressBar.maxProgressBar()
            self.__theSimpleWaitProgressBar.hide()
            
        else:
            # if NOT discretizing to black/white, the "image sequence array" values are to be used:

            # show a panel containing a progress bar:        
            self.__theSimpleWaitProgressBar.setTitleTextRange("cdImageLayer: Computing edge detection on images.", " ", 0, self.theCurrentImage.height())
            self.__theSimpleWaitProgressBar.show()
    
            # set the flag for the current edge array in the sequence as False, as we're computing it now:
            self.extractedCellDataNPIsReadyFlags[self.__theCurrentIndex] = False
    
            xmask, ymask = self.get_prewitt_masks() 
    
            width = self.theCurrentImage.width()
            height = self.theCurrentImage.height()
    
            # create a new greyscale image for the output 
    #         outimg = Image.new('L', (width, height)) 
    #         outpixels = list(outimg.getdata()) 
         
            for y in xrange(height): 
    
                self.__theSimpleWaitProgressBar.setValue(y)
    
                for x in xrange(width): 
                    sumX, sumY, magnitude = 0, 0, 0 
         
                    if y == 0 or y == height-1: magnitude = 0 
                    elif x == 0 or x == width-1: magnitude = 0 
                    else: 
                        for i in xrange(-1, 2): 
                            for j in xrange(-1, 2): 
                                lX = x+i
                                lY = y+j
                                # convolve the image pixels with the Prewitt mask,
                                #     approximating dI/dx 
                                r = int(self.imageNPArray[self.__theCurrentIndex, lY, lX, 0])
                                g = int(self.imageNPArray[self.__theCurrentIndex, lY, lX, 1])
                                b = int(self.imageNPArray[self.__theCurrentIndex, lY, lX, 2])
                                gray = (r + g + b) / 3
                                sumX += (gray) * xmask[i+1, j+1] 
         
                        for i in xrange(-1, 2): 
                            for j in xrange(-1, 2): 
                                lX = x+i
                                lY = y+j
                                # convolve the image pixels with the Prewitt mask,
                                #     approximating dI/dy 
                                r = int(self.imageNPArray[self.__theCurrentIndex, lY, lX, 0])
                                g = int(self.imageNPArray[self.__theCurrentIndex, lY, lX, 1])
                                b = int(self.imageNPArray[self.__theCurrentIndex, lY, lX, 2])
                                gray = (r + g + b) / 3
                                sumY += (gray) * ymask[i+1, j+1] 
         
                    # approximate the magnitude of the gradient 
                    magnitude = abs(sumX) + abs(sumY)
         
                    if magnitude > 255: magnitude = 255 
                    if magnitude < 0: magnitude = 0 
    
                    self.extractedCellDataNPArray[self.__theCurrentIndex, y, x, 0] = numpy.uint8 (255 - magnitude)
                    self.extractedCellDataNPArray[self.__theCurrentIndex, y, x, 1] = numpy.uint8 (255 - magnitude)
                    self.extractedCellDataNPArray[self.__theCurrentIndex, y, x, 2] = numpy.uint8 (255 - magnitude)
    
            # set the flag for the current edge array in the sequence as True, as we're computing it now:
            self.extractedCellDataNPIsReadyFlags[self.__theCurrentIndex] = True
    
            self.__theSimpleWaitProgressBar.maxProgressBar()
            self.__theSimpleWaitProgressBar.hide()
        
        # end         if ( self.getAProcessingModeStatusForImageNP(CDConstants.ImageNPUseDiscretizedToBWMode) == True )

        CDConstants.printOut( "    - DEBUG ----- CDImageNP.__theTrueComputeCurrentEdge(): done.", CDConstants.DebugExcessive )

    # end of   def __theTrueComputeCurrentEdge(self)
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # 2011 - Mitja: normalizeAllImages() images theImage from the current layer in the sequence:
    #    
    # DO NOT USE: the following suggestion thrashes the system!!!!... 2GB of VM dump on disk and stuck:
    #     
    #         # convert to floats in the range [0.0, 1.0] :
    #         f = (self.imageNPArray - self.imageNPArray.min()) \
    #             / float(self.imageNPArray.max() - self.imageNPArray.min())
    #     
    #         # convert back to bytes:
    #         self.imageNPArray = (f * 255).astype(numpy.uint8)
    # 
    # ------------------------------------------------------------------   
    def normalizeAllImages(self):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.normalizeAllImages(): starting.", CDConstants.DebugExcessive )

        # do nothing if the current array size isn't anything:
        if (self.__sizeX <= 0) or (self.__sizeY <= 0) or (self.__sizeZ <= 0) :
            return

        # do nothing if the current image index doesn't have a corresponding image:
        if (self.__theCurrentIndex < 0) or (self.__theCurrentIndex >= self.__sizeZ) :
            return


        lTheMaxImageNPValue = self.imageNPArray.max()
        CDConstants.printOut( "___ - DEBUG ----- CDImageNP.normalizeAllImages() lTheMaxImageNPValue == "+str(lTheMaxImageNPValue) , CDConstants.DebugVerbose )


        if (lTheMaxImageNPValue >= 1) and (lTheMaxImageNPValue < 128):

            if (lTheMaxImageNPValue < 2) :
                lTmpArray = numpy.left_shift( self.imageNPArray, 7)
                CDConstants.printOut( " imageNPArray 7 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxImageNPValue < 4) :
                lTmpArray = numpy.left_shift( self.imageNPArray, 6)
                CDConstants.printOut( " imageNPArray 6 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxImageNPValue < 8) :
                lTmpArray = numpy.left_shift( self.imageNPArray, 5)
                CDConstants.printOut( " imageNPArray 5 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxImageNPValue < 16) :
                lTmpArray = numpy.left_shift( self.imageNPArray, 4)
                CDConstants.printOut( " imageNPArray 4 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxImageNPValue < 32) :
                lTmpArray = numpy.left_shift( self.imageNPArray, 3)
                CDConstants.printOut( " imageNPArray 3 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxImageNPValue < 64) :
                lTmpArray = numpy.left_shift( self.imageNPArray, 2)
                CDConstants.printOut( " imageNPArray 2 bit shift done " , CDConstants.DebugExcessive )
            else :   # i.e. (lTheMaxImageNPValue < 128) :
                lTmpArray = numpy.left_shift( self.imageNPArray, 1)
                CDConstants.printOut( " imageNPArray 1 bit shift done " , CDConstants.DebugExcessive )

            self.imageNPArray = lTmpArray

            lTheMaxImageNPValue = self.imageNPArray.max()
            CDConstants.printOut( "___ - DEBUG ----- CDImageNP.normalizeAllImages() now new lTheMaxImageNPValue == "+str(lTheMaxImageNPValue) , CDConstants.DebugVerbose )

        # end of  if (lTheMaxImageNPValue >= 1) and (lTheMaxImageNPValue < 128)



        # discretizing to black/white, i.e. the "volume array" values are to be first set to either 0 or 1:
        lTmpZerosLikeArray = numpy.zeros_like( self.imageNPArray )
        lTmpBoolArray = numpy.not_equal (self.imageNPArray , lTmpZerosLikeArray)
        self.discretizedNPArray = lTmpBoolArray.astype (numpy.uint8)

        lTheMaxVolumeArrayValue = self.discretizedNPArray.max()
        CDConstants.printOut( "___ - DEBUG ----- CDImageNP.normalizeAllImages() lTheMaxVolumeArrayValue == "+str(lTheMaxVolumeArrayValue) , CDConstants.DebugVerbose )


        if (lTheMaxVolumeArrayValue >= 1) and (lTheMaxVolumeArrayValue < 128):

            if (lTheMaxVolumeArrayValue < 2) :
                lTmpArray = numpy.left_shift( self.discretizedNPArray, 7)
                CDConstants.printOut( " discretizedNPArray 7 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxVolumeArrayValue < 4) :
                lTmpArray = numpy.left_shift( self.discretizedNPArray, 6)
                CDConstants.printOut( " discretizedNPArray 6 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxVolumeArrayValue < 8) :
                lTmpArray = numpy.left_shift( self.discretizedNPArray, 5)
                CDConstants.printOut( " discretizedNPArray 5 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxVolumeArrayValue < 16) :
                lTmpArray = numpy.left_shift( self.discretizedNPArray, 4)
                CDConstants.printOut( " discretizedNPArray 4 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxVolumeArrayValue < 32) :
                lTmpArray = numpy.left_shift( self.discretizedNPArray, 3)
                CDConstants.printOut( " discretizedNPArray 3 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxVolumeArrayValue < 64) :
                lTmpArray = numpy.left_shift( self.discretizedNPArray, 2)
                CDConstants.printOut( " discretizedNPArray 2 bit shift done " , CDConstants.DebugExcessive )
            else :   # i.e. (lTheMaxVolumeArrayValue < 128) :
                lTmpArray = numpy.left_shift( self.discretizedNPArray, 1)
                CDConstants.printOut( " discretizedNPArray 1 bit shift done " , CDConstants.DebugExcessive )

            self.discretizedNPArray = lTmpArray

            lTheMaxVolumeArrayValue = self.discretizedNPArray.max()
            CDConstants.printOut( "___ - DEBUG ----- CDImageNP.normalizeAllImages() now new lTheMaxVolumeArrayValue == "+str(lTheMaxVolumeArrayValue) , CDConstants.DebugVerbose )

        # end of  if (lTheMaxVolumeArrayValue >= 1) and (lTheMaxVolumeArrayValue < 128)



# 
# 
#         # show a panel containing a progress bar:    
#         lProgressBarPanel=CDWaitProgressBar("Test normalizeAllImages array x="+str(self.__sizeX)+" y="+str(self.__sizeY)+"  z="+str(self.__sizeZ), 100)
#         lProgressBarPanel.show()
#         lProgressBarPanel.setRange(0, self.__sizeZ)
# 
# 
#         # for each image in sequence:
#         for k in xrange(0, self.__sizeZ, 1):
#             # for each i,j point in an image:
#             for i in xrange(0, self.__sizeX, 1):
#                 for j in xrange(0, self.__sizeY, 1):
#                     if  (lTmpBoolArray[k, j, i, 0] == True) or (lTmpBoolArray[k, j, i, 1] == True) or (lTmpBoolArray[k, j, i, 2] == True) :
#                         CDConstants.printOut(  "at i,j,k ["+str(i)+","+str(j)+","+str(k)+"] self.imageNPArray[k,j,i] = "+str(self.imageNPArray[k, j, i])+\
#                             "  lTmpBoolArray[k,j,i] = "+str(lTmpBoolArray[k, j, i])+"  self.discretizedNPArray[k,j,i] = "+str(self.discretizedNPArray[k,j,i]), CDConstants.DebugExcessive )
# 
# #                     if  (self.imageNPArray[k, j, i, 0] > 0) or (self.imageNPArray[k, j, i, 1] > 0) or (self.imageNPArray[k, j, i, 2] > 0) :
# #                         print "yaaa!"
# #                         print "yaaa!"
# #                         print "yaaa!"
# #                         print "yaaa!"
# #                         print "yaaa!"
# #                         print "yaaa!"
# #                         print "yaaa!"
# #                         print "yaaa!"
# #                         print "yaaa!"
# #                         print "yaaa!"
# #                         print "yaaa!"
# 
# #                        print "i,j,k [",i,j,k,"] =", self.imageNPArray[j, i, k]
# #                     print "-----------------------------------"
# #                 print "===================================="
#                     
#             lProgressBarPanel.setValue(k)
# 
#         # hide the panel containing a progress bar:
#         lProgressBarPanel.maxProgressBar()
#         lProgressBarPanel.accept()




#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "type(self.imageNPArray) = ", type(self.imageNPArray)
#         print "type(self.imageNPArray[0,0,0,0]) = ", type(self.imageNPArray[0,0,0,0]), " and contains ", self.imageNPArray[0,0,0,0]
#         print "self.imageNPArray.max() = ", self.imageNPArray.max()
#         print "self.imageNPArray.min() = ", self.imageNPArray.min()
# 
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "type(lTmpZerosLikeArray) = ", type(lTmpZerosLikeArray)
#         print "type(lTmpZerosLikeArray[0,0,0,0]) = ", type(lTmpZerosLikeArray[0,0,0,0]), " and contains ", lTmpZerosLikeArray[0,0,0,0]
#         print "lTmpZerosLikeArray.max() = ", lTmpZerosLikeArray.max()
#         print "lTmpZerosLikeArray.min() = ", lTmpZerosLikeArray.min()
# 
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
# 
#         print "type(lTmpBoolArray) = ", type(lTmpBoolArray)
#         print "type(lTmpBoolArray[0,0,0,0]) = ", type(lTmpBoolArray[0,0,0,0]), " and contains ", lTmpBoolArray[0,0,0,0]
#         print "lTmpBoolArray.max() = ", lTmpBoolArray.max()
#         print "lTmpBoolArray.min() = ", lTmpBoolArray.min()
#         # print lTmpBoolArray
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
# 
#         print "type(self.discretizedNPArray) = ", type(self.discretizedNPArray)
#         print "type(self.discretizedNPArray[0,0,0,0]) = ", type(self.discretizedNPArray[0,0,0,0]), " and contains ", self.discretizedNPArray[0,0,0,0]
#         print "self.discretizedNPArray.max() = ", self.discretizedNPArray.max()
#         print "self.discretizedNPArray.min() = ", self.discretizedNPArray.min()
#         # print self.discretizedNPArray
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"

        CDConstants.printOut( "    - DEBUG ----- CDImageNP.normalizeAllImages(): done.", CDConstants.DebugExcessive )

    # end of     def normalizeAllImages(self)
    # ------------------------------------------------------------------   
    # ------------------------------------------------------------------   





    # ------------------------------------------------------------------
    # 2011 - Mitja: setCurrentImageAndNPArrayLayer() loads a new
    #     QImage into self.theCurrentImage, and sets pixel values
    #     in one layer in the numpy array from self.theCurrentImage:
    # ------------------------------------------------------------------   
    def setCurrentImageAndNPArrayLayer(self, pImage):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.setCurrentImageAndNPArrayLayer(): starting.", CDConstants.DebugExcessive )

        # skip parameters that are not proper QImage instances:
        if isinstance( pImage,  QtGui.QImage ) == False:
            return

        lTmpPixmap = QtGui.QPixmap.fromImage(pImage)
        lTmpArrayBGRA = self.qimage2numpy(QtGui.QImage(lTmpPixmap.toImage()), "array")

        # set the flag for the current image array in the sequence as True, as we've loaded it now:
        self.imageInNPIsReadyFlags[self.__theCurrentIndex] = True
        # set the flag for the other related arrays as False, as we've just modified the original:
        self.extractedCellDataNPIsReadyFlags[self.__theCurrentIndex] = True


        pass # 154 prrint "CDImageNP.setCurrentImageAndNPArrayLayer() - self.imageNPArray[self.__theCurrentIndex].shape =",self.imageNPArray[self.__theCurrentIndex].shape
        pass # 154 prrint "CDImageNP.setCurrentImageAndNPArrayLayer() - lTmpArrayBGRA.shape =",lTmpArrayBGRA.shape

        self.imageNPArray[self.__theCurrentIndex] = lTmpArrayBGRA

        # obtain the current image data from one layer in the image numpy array 
        lTmpOneImageArray = self.imageNPArray[self.__theCurrentIndex]
        self.theCurrentImage = self.rgb2qimage(lTmpOneImageArray)

        CDConstants.printOut( "___ - DEBUG ----- CDImageNP.setCurrentImageAndNPArrayLayer() self.__theCurrentIndex == "+str(self.__theCurrentIndex)+" done." , CDConstants.DebugVerbose )

    # ------------------------------------------------------------------









    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def theTrueExtractCells(self): 
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.theTrueExtractCells(): starting.", CDConstants.DebugExcessive )

        return


        # ------------------------------------------------------------------
        # if NOT discretizing to black/white, the "image sequence array" values are to be used:
        # ------------------------------------------------------------------

        lXmask, lYmask = self.get_prewitt_masks()

        lWidth = self.__sizeX
        lHeight = self.__sizeY
        lDepth = self.__sizeZ

        # show a panel containing a progress bar:        
        self.__theProgressBarWithImage=CDWaitProgressBarWithImage("Extracting Cell Areas from Image.", self.theCurrentImage.height())
        self.__theProgressBarWithImage.show()
        self.__theProgressBarWithImage.setRange(0, self.theCurrentImage.height())

        lPixmap = QtGui.QPixmap( lDepth, lHeight)
        lPixmap.fill(QtCore.Qt.transparent)

        # store the pixmap holding the specially rendered scene:
        self.__theProgressBarWithImage.setImagePixmap(lPixmap)
        self.__theProgressBarWithImage.__theProgressBarImageLabel.image = lPixmap.toImage()   
        self.__theProgressBarWithImage.__theProgressBarImageLabel.width = int( lPixmap.width() )
        self.__theProgressBarWithImage.__theProgressBarImageLabel.height = int ( lPixmap.height() )

        if ( self.__theProgressBarWithImage.__theContentWidget.width() < (lDepth + 20) ):
            lTheNewWidth = lDepth + 20
        else:
            lTheNewWidth = self.__theProgressBarWithImage.__theContentWidget.width()
        if ( self.__theProgressBarWithImage.__theContentWidget.height() < (lHeight + 20) ):
            lTheNewHeight = lHeight + 20
        else:
            lTheNewHeight = self.__theProgressBarWithImage.__theContentWidget.height()
        self.__theProgressBarWithImage.__theContentWidget.resize(lTheNewWidth, lTheNewHeight) #asdf 
        self.__theProgressBarWithImage.__theContentWidget.update()
        self.__theProgressBarWithImage.adjustSize()

        # -------------------------------
        # scan across x-direction layers:
        # -------------------------------
        for x in xrange(lWidth): 

            # prepare a QPainter for visual feedback:
            lTmpPainter = QtGui.QPainter(lPixmap)
            lTmpPen = QtGui.QPen(QtCore.Qt.black)
            lTmpPen.setWidth(2)
            lTmpPen.setCosmetic(True)
            lTmpPainter.setPen(lTmpPen)
            lTmpBrush = QtGui.QBrush(QtGui.QColor(QtCore.Qt.red))
            lTmpPainter.setBrush(lTmpBrush)

            self.__theProgressBarWithImage.setTitle( self.tr(" Scanning [x] layer %1 of %2 from Image Sequence Volume \n to generate 3D Contour-boundary points... ").arg( \
                str(x) ).arg( str(lWidth) )  )

            CDConstants.printOut( "___ - DEBUG ----- CDImageNP.self.theTrueExtractCells() - lPixmap w,h =" + \
                  str(self.__theProgressBarWithImage.__theProgressBarImageLabel.width) + " " + str(self.__theProgressBarWithImage.__theProgressBarImageLabel.height) + \
                  " Scanning [x] layer "+str(x)+" of "+str(lWidth)+" from Image Sequence Volume to generate 3D Contour-boundary points.", CDConstants.DebugVerbose )

            # adjusts the size of the label widget to fit its contents (i.e. the pixmap):
            self.__theProgressBarWithImage.__theProgressBarImageLabel.adjustSize()
            self.__theProgressBarWithImage.__theProgressBarImageLabel.show()
            self.__theProgressBarWithImage.__theProgressBarImageLabel.update()

            # -----------------------------
            # scan across y-direction rows:
            # -----------------------------
            for y in xrange(lHeight): 

                # provide visual feedback to user:
                self.__theProgressBarWithImage.setValue(y)
                QtGui.QApplication.processEvents()

                # --------------------------------
                # scan across z-direction columns:
                # --------------------------------
                for z in xrange(lDepth): 
                    sumZ, sumY, magnitude = 0, 0, 0 

                    if y == 0 or y == lHeight-1: magnitude = 0 
                    elif z == 0 or z == lDepth-1: magnitude = 0 
                    else: 
                        for k in xrange(-1, 2): 
                            for j in xrange(-1, 2): 
                                lZ = z+k
                                lY = y+j
                                # convolve the image pixels with the Prewitt mask,
                                #     approximating dI/dz 
                                r = int(self.imageSequenceArray[lZ, lY, x, 0])
                                g = int(self.imageSequenceArray[lZ, lY, x, 1])
                                b = int(self.imageSequenceArray[lZ, lY, x, 2])
                                gray = (r + g + b) / 3
                                sumZ += (gray) * lXmask[k+1, j+1] 
         
                        for k in xrange(-1, 2): 
                            for j in xrange(-1, 2): 
                                lZ = z+k
                                lY = y+j
                                # convolve the image pixels with the Prewitt mask,
                                #     approximating dI/dy 
                                r = int(self.imageSequenceArray[lZ, lY, x, 0])
                                g = int(self.imageSequenceArray[lZ, lY, x, 1])
                                b = int(self.imageSequenceArray[lZ, lY, x, 2])
                                gray = (r + g + b) / 3
                                sumY += (gray) * lYmask[k+1, j+1] 
         
                    # approximate the magnitude of the gradient 
                    magnitude = abs(sumZ) + abs(sumY)
         
                    if magnitude > 255: magnitude = 255 
                    if magnitude < 0: magnitude = 0 
    
                    self.contoursSequenceArray[z, y, x, 0] = numpy.uint8 (255 - magnitude)
                    self.contoursSequenceArray[z, y, x, 1] = numpy.uint8 (255 - magnitude)
                    self.contoursSequenceArray[z, y, x, 2] = numpy.uint8 (255 - magnitude)

                    # provide some visual feedback to user by drawing the currently processed pixel:
#                         lTmpPainter = QtGui.QPainter(lPixmap)
                    lTmpColor = QtGui.QColor( (255 - magnitude), (255 - magnitude), (255 - magnitude) )
                    lTmpPen = QtGui.QPen()
                    lTmpPen.setColor(lTmpColor)
                    lTmpPainter.setPen(lTmpPen)
                    lTmpPainter.drawPoint(z,y)

                # -------------------------------------------
                # <-- end of scan across z-direction columns.
                # -------------------------------------------

            # ----------------------------------------
            # <-- end of scan across y-direction rows.
            # ----------------------------------------

            lTmpPainter.end()
            # provide visual feedback to user:
            self.__theProgressBarWithImage.__theProgressBarImageLabel.drawPixmapAtPoint(lPixmap)
            self.__theProgressBarWithImage.__theProgressBarImageLabel.update()


        # ------------------------------------------
        # <-- end of scan across x-direction layers.
        # ------------------------------------------
    
        # set the flag for the contours array in the sequence as True, as we've just computed it:
        self.contoursAreReadyFlag = True
    
        self.__theProgressBarWithImage.maxProgressBar()
#         self.__theProgressBarWithImage.accept()
        self.__theProgressBarWithImage.hide()

        CDConstants.printOut( "    - DEBUG ----- CDImageNP.theTrueExtractCells(): done.", CDConstants.DebugExcessive )

    # end of   def theTrueExtractCells(self)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------









    # ------------------------------------------------------------------
    # 2011 - Mitja: the setNPLoadedFromFiles() function is to mimic
    #    the behavior of the CDImageLayer class:
    # ------------------------------------------------------------------   
    def setNPLoadedFromFiles(self, pTrueOrFalse):
        self.__imageNPLoaded = pTrueOrFalse
        if isinstance( self._graphicsSceneWidget, QtGui.QWidget ) == True:
            self._graphicsSceneWidget.scene.update()
        CDConstants.printOut( "___ - DEBUG ----- CDImageNP.setNPLoadedFromFiles( " + str(self.__imageNPLoaded) + " )", CDConstants.DebugVerbose )




    # ------------------------------------------------------------------
    # 2011 - Mitja: setCurrentIndexWithoutUpdatingGUI() is to set the current image index,
    #   during importing images and other non-GUI tasks:
    # ------------------------------------------------------------------   
    def setCurrentIndexWithoutUpdatingGUI(self, pValue):
        self.__theCurrentIndex = pValue

        CDConstants.printOut( "___ - DEBUG ----- CDImageNP.setCurrentIndexWithoutUpdatingGUI() self.__theCurrentIndex == "+str(self.__theCurrentIndex) , CDConstants.DebugVerbose )
    # end of    def setCurrentIndexWithoutUpdatingGUI(self, pValue)
    # ------------------------------------------------------------------   




    # ------------------------------------------------------------------
    # 2011 - Mitja: setCurrentIndexInImageNP() is to set the current image index within the array stack:
    # ------------------------------------------------------------------   
    def setCurrentIndexInImageNP(self, pValue):
        self.__theCurrentIndex = pValue

        CDConstants.printOut(  "___ - DEBUG ----- CDImageNP.setCurrentIndexInImageNP()  --  1.   self.__theCurrentIndex=="+str(self.__theCurrentIndex), CDConstants.DebugTODO )

        #  create images... theCurrentImage and theCurrentExtractedCellDataImage and theCurrentDiscretizedImage from the current layer in the sequence arrays:
        # these images are now *not* painted here, but from setCurrentIndexInImageNP() ...
        # ... should *not* call imageCurrentImageNP() from paintTheImageNP(),
        #    because paintTheImageNP() is part of the repainting and ought not open additional widgets or cause repaints...
        #    (and imageCurrentImageNP() may open dialog boxes etc.)
        self.imageCurrentImageNP()

        CDConstants.printOut(  "___ - DEBUG ----- CDImageNP.setCurrentIndexInImageNP()  --  2.   calling self.imageCurrentImageNP() DONE", CDConstants.DebugTODO )

        # emit a signal to update image sequence size GUI controls:
        lDict = { \
            0: str(self.__theCurrentIndex), \
            1: str(self.imageNPFileNames[self.__theCurrentIndex]), \
            }

        self.signalThatCurrentIndexSet.emit(lDict)

        CDConstants.printOut(  "___ - DEBUG ----- CDImageNP.setCurrentIndexInImageNP()  --  3.   self.signalThatCurrentIndexSet.emit( lDict=="+str(lDict)+" )", CDConstants.DebugTODO )

        CDConstants.printOut(  "___ - DEBUG ----- CDImageNP.setCurrentIndexInImageNP() self.__theCurrentIndex == "+str(self.__theCurrentIndex) , CDConstants.DebugVerbose )
    # end of    def setCurrentIndexInImageNP(self, pValue)
    # ------------------------------------------------------------------   



    # ------------------------------------------------------------------
    # 2011 - Mitja: resetNPDimensions() is to set x,y,z,channels for image processing:
    # ------------------------------------------------------------------   
    def resetNPDimensions(self, pX, pY, pZ, pChannels):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.resetNPDimensions( pX="+str(pX)+", pY="+str(pY)+", pZ="+str(pZ)+", pChannels="+str(pChannels)+" ) starting.", CDConstants.DebugExcessive )

        if (pX > 0):
            self.__sizeX = pX
        else:
            self.__sizeX = 1
        if (pY > 0):
            self.__sizeY = pY
        else:
            self.__sizeY = 1
        if (pZ > 0):
            self.__sizeZ = pZ
        else:
            self.__sizeZ = 1
        if (pChannels > 0):
            self.__cChannels = pChannels
        else:
            self.__cChannels = 1

        # a fresh numpy-based array - the four dimensions are:
        #   z = image layers,  y = height,  x = width, [b g r], even if image may be [a b g r]
        #   i.e. indexed from the slowest (most distant) to the fastest (closest data to each other):

        self.imageNPArray = numpy.zeros( \
            (self.__sizeZ,  self.__sizeY,  self.__sizeX,  3), \
            dtype=numpy.uint8  )
        self.extractedCellDataNPArray = numpy.zeros( \
            (self.__sizeZ,  self.__sizeY,  self.__sizeX,  3), \
            dtype=numpy.uint8  )
        self.discretizedNPArray = numpy.zeros( \
            (self.__sizeZ,  self.__sizeY,  self.__sizeX,  3), \
            dtype=numpy.uint8  )

        # reset to fresh and empty (all False) flag arrays as well:
        self.imageInNPIsReadyFlags = numpy.zeros( (self.__sizeZ), dtype=numpy.bool )
        self.extractedCellDataNPIsReadyFlags = numpy.zeros( (self.__sizeZ), dtype=numpy.bool )
        self.contoursAreReadyFlag = False

        # reset to fresh and empty (all " ") filename strings as well:
        for i in xrange(self.__sizeZ):
            self.imageNPFileNames[i] = " "

        # emit a signal to update image size GUI controls:
        if ( int(self.__sizeZ) == 1 ) :
            lLabel = "image"
        else:
            lLabel = "images"

        lDict = { \
            0: str(int(self.__sizeX)), \
            1: str(int(self.__sizeY)), \
            2: str(int(self.__sizeZ)), \
            3: str(lLabel) \
            }

        self.signalThatImageNPResized.emit(lDict)

        CDConstants.printOut( "___ - DEBUG ----- CDImageNP.resetNPDimensions() self.__sizeX,Y,Z, self.__cChannels == "+str(self.__sizeX)+" "+str(self.__sizeY)+" "+str(self.__sizeZ)+" "+str(self.__cChannels) , CDConstants.DebugVerbose )




        #   an empty array, into which to write pixel values,
        #   one Z layer for each image in the sequence,
        #   and we use numpy.int32 as data type to hold RGBA values:
        # self.imageNPArray = numpy.zeros( (self.__sizeY, self.__sizeX, self.__sizeZ), dtype=numpy.int )


#         TODO: this testing only necessary when NO image sequence loaded:

        # if there is no image sequence loaded, then do nothing after clearing the array:



#         if  (self.__imageNPLoaded == False):
#     
#             # set test array content:
# 
#             # show a panel containing a progress bar:    
#             lProgressBarPanel=CDWaitProgressBar("Test content image sequence array x="+str(self.__sizeX)+" y="+str(self.__sizeY)+"  z="+str(self.__sizeZ), 100)
#             lProgressBarPanel.show()
#             lProgressBarPanel.setRange(0, self.__sizeZ)
# 
# 
#             # for each image in sequence:
#             for k in xrange(0, self.__sizeZ, 1):
#                 # for each i,j point in an image:
#                 for i in xrange(0, self.__sizeX, 1):
#                     for j in xrange(0, self.__sizeY, 1):
#                         if (i == j) and (j == k):
#                             self.imageNPArray[k, j, i, 0] = numpy.uint8 (127)
#                         else:
#                             self.imageNPArray[k, j, i, 0] = numpy.uint8 ( 0 )
# #                        print "i,j,k [",i,j,k,"] =", self.imageNPArray[j, i, k]
# #                     print "-----------------------------------"
# #                 print "===================================="
#                         
#                 lProgressBarPanel.setValue(k)
# 
#             # close the panel containing a progress bar:
#             lProgressBarPanel.maxProgressBar()
#             lProgressBarPanel.accept()
# 
#             self.normalizeAllImages()

    # end of  def resetNPDimensions(self, pX, pY, pZ)
    # ------------------------------------------------------------------   


    # ------------------------------------------------------------------
    # 2011 - Mitja: resetToOneProcessingModeForImageNP()
    #    sets (to 1 AKA True) a binary-flag in a class global keeping track of the modes
    #    for generating PIFF from displayed imported image
    # ------------------------------------------------------------------
    def resetToOneProcessingModeForImageNP(self, pValue):

        # if we are changing choice on discretization to BW mode,  invalidate all computed edges and contours:
        if (    (pValue == CDConstants.ImageNPUseDiscretizedToBWMode) and \
                (self.getAProcessingModeStatusForImageNP(CDConstants.ImageNPUseDiscretizedToBWMode) == False)  ) \
            or \
            (   (pValue != CDConstants.ImageNPUseDiscretizedToBWMode) and \
                (self.getAProcessingModeStatusForImageNP(CDConstants.ImageNPUseDiscretizedToBWMode) == True)  ):
        # if we are changing choice on discretization to BW mode,  invalidate all computed edges and contours:
            self.extractedCellDataNPIsReadyFlags = numpy.zeros( (self.__sizeZ), dtype=numpy.bool )
            self.contoursAreReadyFlag = False

        # set bitwise pValue, to set only the specific bit to 1 and all other bits to 0:
        self.__theProcessingModeForImageNP = (1 << pValue)

        # bin() does not exist in Python 2.5:
        if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
            CDConstants.printOut( "___ - DEBUG ----- CDImageNP.resetToOneProcessingModeForImageNP() bin(self.__theProcessingModeForImageNP) == "+str(bin(self.__theProcessingModeForImageNP))+" from pValue =="+str(pValue) , CDConstants.DebugVerbose )
        else:
            CDConstants.printOut( "___ - DEBUG ----- CDImageNP.resetToOneProcessingModeForImageNP() self.__theProcessingModeForImageNP == "+str(self.__theProcessingModeForImageNP)+" from pValue =="+str(pValue) , CDConstants.DebugVerbose )

    # end of  def resetToOneProcessingModeForImageNP(self, pValue)
    # ------------------------------------------------------------------   



    # ------------------------------------------------------------------
    # 2011 - Mitja: getAProcessingModeStatusForImageNP()
    #    returns a Boolean from the binary-flag in a class global keeping track of the modes
    #    for generating PIFF from displayed imported image sequence
    # ------------------------------------------------------------------   
    def getAProcessingModeStatusForImageNP(self, pValue):
        if ( self.__theProcessingModeForImageNP & (1 << pValue) ):
            # bin() does not exist in Python 2.5:
            if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
                CDConstants.printOut( "___ - DEBUG ----- CDImageNP.getAProcessingModeStatusForImageNP() TRUE bin(pValue, (1 << pValue)) == "+str(pValue)+" , "+str(bin(1 << pValue)) , CDConstants.DebugVerbose )
            else:
                CDConstants.printOut( "___ - DEBUG ----- CDImageNP.getAProcessingModeStatusForImageNP() TRUE (pValue, (1 << pValue) == "+str(pValue)+" , "+str(1 << pValue) , CDConstants.DebugVerbose )
            return True
        else:
            # bin() does not exist in Python 2.5:
            if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
                CDConstants.printOut( "___ - DEBUG ----- CDImageNP.getAProcessingModeStatusForImageNP() FALSE bin(pValue, (1 << pValue)) == "+str(pValue)+" , "+str(bin(1 << pValue)) , CDConstants.DebugVerbose )
            else:
                CDConstants.printOut( "___ - DEBUG ----- CDImageNP.getAProcessingModeStatusForImageNP() FALSE pValue, (1 << pValue) == "+str(pValue)+" , "+str(1 << pValue) , CDConstants.DebugVerbose )
            return False




    # ------------------------------------------------------------------
    # 2011 - Mitja: getNPCurrentColor():
    # ------------------------------------------------------------------   
    def getNPCurrentColor(self):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.getNPCurrentColor returning  theImageNPColor=="+str(self.__theImageNPColor)+" done.", CDConstants.DebugExcessive )
        return (self.__theImageNPColor)



    # ------------------------------------------------------------------
    # 2011 - Mitja: getNPWallColor():
    # ------------------------------------------------------------------   
    def getNPWallColor(self):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.getNPWallColor returning  theImageNPWallColor=="+str(self.__theImageNPWallColor)+" done.", CDConstants.DebugExcessive )
        return (self.__theImageNPWallColor)





    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    #                       from qimage2ndarray.py :
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    
    
    
    
    """QImage <-> numpy.ndarray conversion module.
    
    *** ATTENTION: This code is outdated - I released a better extension
    *** named 'qimage2ndarray' (that completes all TODO items below) in
    *** the meantime, which is available via PyPI and here:
    *** http://kogs-www.informatik.uni-hamburg.de/~meine/software/qimage2ndarray/
    
    This supports conversion in both directions; note that in contrast to
    C++, in Python it is not possible to convert QImages into ndarrays
    without copying the data.  The conversion functions in the opposite
    direction however do not copy the data.
    
    TODO:
    - support record arrays in rgb2qimage
      (i.e. grok the output of qimage2numpy)
    - support unusual widths/alignments also in gray2qimage and
      rgb2qimage
    - make it possible to choose between views and copys of the data
      (eventually in both directions, when implemented in C++)
    - allow for normalization in numpy->QImage conversion
      (i.e. to quickly visualize images with different value ranges)
    - implement in C++
    """
    
    # import numpy
    # from PyQt4.QtGui import QImage, QColor

    bgra_dtype = numpy.dtype({'b': (numpy.uint8, 0),
                              'g': (numpy.uint8, 1),
                              'r': (numpy.uint8, 2),
                              'a': (numpy.uint8, 3)})

    def qimage2numpy(self, pImageIn, pDataTypeOut = 'array'):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.qimage2numpy starting.", CDConstants.DebugExcessive )
        """Convert QImage to numpy.ndarray.  The pDataTypeOut defaults to uint8
        for QImage.Format_Indexed8 or `bgra_dtype` (i.e. a record array)
        for 32bit color images.  You can pass a different pDataTypeOut to use, or
        'array' to get a 3D uint8 array for color images."""
        result_shape = (pImageIn.height(), pImageIn.width())
        temp_shape = (pImageIn.height(),
                      pImageIn.bytesPerLine() * 8 / pImageIn.depth())
        # from Qt documentation:
        #  ... the image depth is the number of bits used to store a single pixel, also called bits per pixel (bpp). The supported depths are 1, 8, 16, 24 and 32.
        if pImageIn.format() in (QtGui.QImage.Format_ARGB32_Premultiplied, \
                                 QtGui.QImage.Format_ARGB32, \
                                 QtGui.QImage.Format_RGB32):
            if pDataTypeOut == 'rec':
                pDataTypeOut = bgra_dtype
            elif pDataTypeOut == 'array':
                # add a 3rd dimension to the array, i.e. 1 entry per color/channel in the image depth: R G B and A:
                pDataTypeOut = numpy.uint8
                pass # 154 prrint "CDImageNP.qimage2numpy() - result_shape = ", result_shape
                pass # 154 prrint "CDImageNP.qimage2numpy() - temp_shape = ", temp_shape
                result_shape += (4, )
                temp_shape += (4, )
                pass # 154 prrint "CDImageNP.qimage2numpy() - pImageIn.format() = ", pImageIn.format()
                pass # 154 prrint "CDImageNP.qimage2numpy() - result_shape = ", result_shape
                pass # 154 prrint "CDImageNP.qimage2numpy() - temp_shape = ", temp_shape
        elif pImageIn.format() == QtGui.QImage.Format_Indexed8:
            pDataTypeOut = numpy.uint8
        else:
            raise ValueError("qimage2numpy only supports 32bit and 8bit images")
        # FIXME: raise error if alignment does not match
        
        # obtain the data buffer starting with the first pixel data,
        #   where bits() is Qt function returning the pointer to data,
        #   and asstring() is the sip.voidptr function returning a Python string:
        buf = pImageIn.bits().asstring(pImageIn.numBytes())
        # obtain a numpy array, where frombuffer() returns
        #   a 1-dimensional array of given datatype objects obtained from buffer
        #   and reshape() gives a new shape to an array without changing its data
        result = numpy.frombuffer(buf, pDataTypeOut).reshape(temp_shape)

        if result_shape != temp_shape:
            pass # 154 prrint "CDImageNP.qimage2numpy() - ======> result_shape != temp_shape, "
            pass # 154 prrint "CDImageNP.qimage2numpy() -         therefore doing:   result = result[:,:result_shape[1]]"
            result = result[:,:result_shape[1]]
        else:
            pass # 154 prrint "CDImageNP.qimage2numpy() - ======> result_shape == temp_shape, "
            pass # 154 prrint "CDImageNP.qimage2numpy() -         therefore  result  stays the same."

        if pImageIn.format() == QtGui.QImage.Format_RGB32 and pDataTypeOut == numpy.uint8:
            pass # 154 prrint "CDImageNP.qimage2numpy() - ======> pImageIn.format() == QtGui.QImage.Format_RGB32 = ", (pImageIn.format() == QtGui.QImage.Format_RGB32)
            pass # 154 prrint "CDImageNP.qimage2numpy() - ======> pDataTypeOut == numpy.uint8 = ", (pDataTypeOut == numpy.uint8)
            pass # 154 prrint "CDImageNP.qimage2numpy() -         therefore doing:   result = result[...,:3]"
            # slice the result so that every 4th byte is dropped, i.e. take 3 bytes at a time:
            result = result[...,:3]
            CDConstants.printOut( "    result[...,0].min, max() = "+str(result[...,0].min()) +" "+ str(result[...,0].max()), CDConstants.DebugExcessive )
            CDConstants.printOut( "    result[...,1].min, max() = "+str(result[...,1].min()) +" "+ str(result[...,1].max()), CDConstants.DebugExcessive )
            CDConstants.printOut( "    result[...,2].min, max() = "+str(result[...,2].min()) +" "+ str(result[...,2].max()), CDConstants.DebugExcessive )
        else:
            pass # 154 prrint "CDImageNP.qimage2numpy() - ======> pImageIn.format() == QtGui.QImage.Format_RGB32 = ", (pImageIn.format() == QtGui.QImage.Format_RGB32)
            pass # 154 prrint "CDImageNP.qimage2numpy() - ======> pDataTypeOut == numpy.uint8 = ", (pDataTypeOut == numpy.uint8)
            pass # 154 prrint "CDImageNP.qimage2numpy() -         therefore  result  stays the same."
            CDConstants.printOut( "    result[...,0].min, max() = "+str(result[...,0].min()) +" "+ str(result[...,0].max()), CDConstants.DebugExcessive )
            CDConstants.printOut( "    result[...,1].min, max() = "+str(result[...,1].min()) +" "+ str(result[...,1].max()), CDConstants.DebugExcessive )
            CDConstants.printOut( "    result[...,2].min, max() = "+str(result[...,2].min()) +" "+ str(result[...,2].max()), CDConstants.DebugExcessive )
            CDConstants.printOut( "    result[...,3].min, max() = "+str(result[...,3].min()) +" "+ str(result[...,3].max()), CDConstants.DebugExcessive )

        CDConstants.printOut( "    - DEBUG ----- CDImageNP.qimage2numpy done.", CDConstants.DebugExcessive )
        
        return result

    def numpy2qimage(self, array):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.numpy2qimage starting.", CDConstants.DebugExcessive )
        if numpy.ndim(array) == 2:
            return self.gray2qimage(array)
        elif numpy.ndim(array) == 3:
            return self.rgb2qimage(array)
        raise ValueError("can only convert 2D or 3D arrays")
    
    def gray2qimage(self, gray):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.gray2qimage starting.", CDConstants.DebugExcessive )
        """Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
        colormap.  The first dimension represents the vertical image axis.
    
        ATTENTION: This QImage carries an attribute `ndimage` with a
        reference to the underlying numpy array that holds the data. On
        Windows, the conversion into a QPixmap does not copy the data, so
        that you have to take care that the QImage does not get garbage
        collected (otherwise PyQt will throw away the wrapper, effectively
        freeing the underlying memory - boom!)."""
        if len(gray.shape) != 2:
            raise ValueError("gray2QImage can only convert 2D arrays")
    
        gray = numpy.require(gray, numpy.uint8, 'C')
    
        h, w = gray.shape
    
        result = QtGui.QImage(gray.data, w, h, QtGui.QImage.Format_Indexed8)
        result.ndarray = gray
        for i in range(256):
            result.setColor(i, QtGui.QColor(i, i, i).rgb())
        return result
    
    # --------------------------
    def rgb2qimage(self, rgb):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.rgb2qimage starting.", CDConstants.DebugExcessive )
        """Convert the 3D numpy array `rgb` into a 32-bit QImage.  `rgb` must
        have three dimensions with the vertical, horizontal and RGB image axes.
    
        ATTENTION: This QImage carries an attribute `ndimage` with a
        reference to the underlying numpy array that holds the data. On
        Windows, the conversion into a QPixmap does not copy the data, so
        that you have to take care that the QImage does not get garbage
        collected (otherwise PyQt will throw away the wrapper, effectively
        freeing the underlying memory - boom!)."""
        if len(rgb.shape) != 3:
            raise ValueError("rgb2QImage only converts 3D arrays")
        if rgb.shape[2] not in (3, 4):
            raise ValueError("rgb2QImage expects the last dimension to contain exactly three (R,G,B) or four (R,G,B,A) channels")
    
        h, w, channels = rgb.shape
    
        # Qt expects 32bit BGRA data for color images:
        bgra = numpy.empty((h, w, 4), numpy.uint8, 'C')
        # originally this was swapping R and B, unclear why?
        # bgra[...,0] = rgb[...,2]
        # bgra[...,1] = rgb[...,1]
        # bgra[...,2] = rgb[...,0]

        # rgb[...,2] = red channel:
        bgra[...,2] = rgb[...,2]
        # rgb[...,1] = green channel:
        bgra[...,1] = rgb[...,1]
        # rgb[...,0] = blue channel:
        bgra[...,0] = rgb[...,0]
        if rgb.shape[2] == 3:
            bgra[...,3].fill(255)
            fmt = QtGui.QImage.Format_RGB32
        else:
            bgra[...,3] = rgb[...,3]
            fmt = QtGui.QImage.Format_ARGB32
    
        result = QtGui.QImage(bgra.data, w, h, fmt)
        result.ndarray = bgra
        return result
    # end of def rgb2qimage(rgb)
    # --------------------------
    
    
    # --------------------------
    # 2012 - Mitja modify to fill alpha channel with transparent where black, and fill only blue where non-black:
    # --------------------------
    def rgb2qimageKtoBandA(self, rgb):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.rgb2qimageKtoBandA starting.", CDConstants.DebugExcessive )
        if len(rgb.shape) != 3:
            raise ValueError("rgb2QImage only converts 3D arrays")
        if rgb.shape[2] not in (3, 4):
            raise ValueError("rgb2QImage expects the last dimension to contain exactly three (R,G,B) or four (R,G,B,A) channels")
    
        h, w, channels = rgb.shape
    
        # Qt expects 32bit BGRA data for color images:
        bgra = numpy.empty((h, w, 4), numpy.uint8, 'C')
        bgra[...,0] = rgb[...,2]
        bgra[...,1] = rgb[...,1]
        bgra[...,2] = rgb[...,0]
    
        # 2012 - Mitja:
    
        # fill the A channel of the original image with all zeros:
        bgra[...,3].fill(0)
        
    #    print "rgb2qimageKtoBandA - bgra = ", bgra
    
        # create two arrays:
        bgraK = numpy.empty((h, w, 4), numpy.uint8, 'C')
        bgraFlags = numpy.zeros((h, w, 4), numpy.uint8, 'C')
    
        # fill bgraFlags as a boolean array with 1-values wherever the input image is not black:
        bgraFlags[bgra!=0] = 1
    
        # fill the B channel with grays from the original image:
        bgraK[...,0] = 80 * (  bgraFlags[...,0] + bgraFlags[...,1] + bgraFlags[...,2]  )
        # fill the R and G channels with zeros, since we want the output to be blue:
        bgraK[...,1].fill(0)
        bgraK[...,2].fill(0)
    
        # make the alpha channel non-zero wherever there is some non-black in the original image:
        #   ...but not totally opaque: about 50% transparency where gray:
        bgraK[...,3] = 40 * (  bgraFlags[...,0] + bgraFlags[...,1] + bgraFlags[...,2]  )
    
    #    print "rgb2qimageKtoBandA - bgraFlags = ", bgraFlags
    #    print "rgb2qimageKtoBandA - bgraK = ", bgraK
    
        fmt = QtGui.QImage.Format_ARGB32
        result = QtGui.QImage(bgraK.data, w, h, fmt)
        result.ndarray = bgraK
        return result
    # --------------------------
    
    
    
    # --------------------------
    # 2012 - Mitja modify to fill alpha channel with transparent where white, and fill only red where black:
    # --------------------------
    def rgb2qimageWtoRandA(self, rgb):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.rgb2qimageWtoRandA starting.", CDConstants.DebugExcessive )
        if len(rgb.shape) != 3:
            raise ValueError("rgb2QImage only converts 3D arrays")
        if rgb.shape[2] not in (3, 4):
            raise ValueError("rgb2QImage expects the last dimension to contain exactly three (R,G,B) or four (R,G,B,A) channels")
    
        h, w, channels = rgb.shape
    
        # Qt expects 32bit BGRA data for color images:
        bgra = numpy.empty((h, w, 4), numpy.uint8, 'C')
        bgra[...,0] = rgb[...,2]
        bgra[...,1] = rgb[...,1]
        bgra[...,2] = rgb[...,0]
    
        # 2012 - Mitja:
    
        # fill the A channel of the original image with all zeros:
        bgra[...,3].fill(0)
        
    #    print "rgb2qimageWtoRandA - bgra = ", bgra
    
        # create two arrays:
        bgraK = numpy.empty((h, w, 4), numpy.uint8, 'C')
        bgraFlags = numpy.zeros((h, w, 4), numpy.uint8, 'C')
    
        # fill bgraFlags as a boolean array with 1-values wherever the input image is non-white:
        bgraFlags[bgra!=255] = 1
    
        # fill the B and G channels with zeros, since we want the output to be red:
        bgraK[...,0].fill(0)
        bgraK[...,1].fill(0)
        # fill the R channel with grays from the original image:
        bgraK[...,2] = 80 * (  bgraFlags[...,0] + bgraFlags[...,1] + bgraFlags[...,2]  )
    
        # make the alpha channel non-zero wherever there is some non-black in the original image:
        #   ...but not totally opaque: about 50% transparency where gray:
        bgraK[...,3] = 40 * (  bgraFlags[...,0] + bgraFlags[...,1] + bgraFlags[...,2]  )
    
    #    print "rgb2qimageWtoRandA - bgraFlags = ", bgraFlags
    #    print "rgb2qimageWtoRandA - bgraK = ", bgraK
    
        fmt = QtGui.QImage.Format_ARGB32
        result = QtGui.QImage(bgraK.data, w, h, fmt)
        result.ndarray = bgraK
        return result
    # --------------------------
    
    
    
    
    # --------------------------
    # 2012 - Mitja modify to fill alpha channel with transparent where white, and fill only red where black:
    # --------------------------
    def rgb2qimageWtoGandA(self, rgb):
        CDConstants.printOut( "    - DEBUG ----- CDImageNP.rgb2qimageWtoGandA starting.", CDConstants.DebugExcessive )
        if len(rgb.shape) != 3:
            raise ValueError("rgb2QImage only converts 3D arrays")
        if rgb.shape[2] not in (3, 4):
            raise ValueError("rgb2QImage expects the last dimension to contain exactly three (R,G,B) or four (R,G,B,A) channels")
    
        h, w, channels = rgb.shape
    
        # Qt expects 32bit BGRA data for color images:
        bgra = numpy.empty((h, w, 4), numpy.uint8, 'C')
        bgra[...,0] = rgb[...,2]
        bgra[...,1] = rgb[...,1]
        bgra[...,2] = rgb[...,0]
    
        # 2012 - Mitja:
    
        # fill the A channel of the original image with all zeros:
        bgra[...,3].fill(0)
        
    #    print "rgb2qimageWtoGandA - bgra = ", bgra
    
        # create two arrays:
        bgraK = numpy.empty((h, w, 4), numpy.uint8, 'C')
        bgraFlags = numpy.zeros((h, w, 4), numpy.uint8, 'C')
    
        # fill bgraFlags as a boolean array with 1-values wherever the input image is non-white:
        bgraFlags[bgra!=255] = 1
    
        # fill the B and R channels with zeros, since we want the output to be red:
        bgraK[...,0].fill(0)
        bgraK[...,2].fill(0)
        # fill the G channel with grays from the original image:
        bgraK[...,1] = 80 * (  bgraFlags[...,0] + bgraFlags[...,1] + bgraFlags[...,2]  )
    
        # make the alpha channel non-zero wherever there is some non-black in the original image:
        #   ...but not totally opaque: about 50% transparency where gray:
        bgraK[...,3] = 40 * (  bgraFlags[...,0] + bgraFlags[...,1] + bgraFlags[...,2]  )
    
    #    print "rgb2qimageWtoGandA - bgraFlags = ", bgraFlags
    #    print "rgb2qimageWtoGandA - bgraK = ", bgraK
    
        fmt = QtGui.QImage.Format_ARGB32
        result = QtGui.QImage(bgraK.data, w, h, fmt)
        result.ndarray = bgraK
        return result
    # --------------------------
    
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    #                       from qimage2ndarray.py done
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    


# end of   class CDImageNP(QtCore.QObject)
# ======================================================================




# ======================================================================
# 2011- Mitja: additional layer to draw an input image for CellDraw
#   on the top of the QGraphicsScene-based cell/region editor
# ======================================================================
# This class draws an image loaded from a file,
#   and it processes mouse click events.
# ======================================================================
class CDImageLayer(QtCore.QObject):
    # --------------------------------------------------------

    # we pass a "dict" parameter with the signalThatMouseMoved parameter, so that
    #   both the mouse x,y coordinates as well as color information can be passed around:

    signalThatMouseMoved = QtCore.pyqtSignal(dict)
    
    # 2010 - Mitja: add functionality for drawing color regions:
    #    where the value for keeping track of what's been changed is:
    #    0 = Color Pick = CDConstants.ImageModePickColor
    #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
    #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon
    #    3 = Extract Cells = CDConstants.ImageModeExtractCells

    # --------------------------------------------------------
    def __init__(self, pParent=None):
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.__init__( pParent == "+str(pParent)+") starting." , CDConstants.DebugExcessive )

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
        self.theImageScaleFactor = 1.0
#
#         # the "thePixmap" is to mimic a QLabel which can be assigned a QPixmap:
#         self.thePixmap = QtGui.QPixmap()
#         # the "processedPixmap" QPixmap is the one we obtained from processedImage:
#         self.processedPixmap = QtGui.QPixmap()

        if isinstance( pParent, QtGui.QWidget ) is True:
            self._graphicsSceneWidget = pParent
        else:
            self._graphicsSceneWidget = None
        CDConstants.printOut( "|^|^|^|^| CDImageLayer.__init__ - self._graphicsSceneWidget == "+str(self._graphicsSceneWidget)+" |^|^|^|^|" , CDConstants.DebugTODO ) 

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

        # the progress bar widget is instantiated in the CellDrawMainWindow class,
        #   and assigned below in setSimpleProgressBarPanel() :
        self.__theSimpleWaitProgressBar = None

        # the progress bar with image widget is instantiated in the CellDrawMainWindow class,
        #   and assigned below in setProgressBarWithImagePanel() :
        self.__theWaitProgressBarWithImage = None

        # 2012 - Mitja: add a separate object to handle NumPy-based image processing:
        self.cdImageNP = CDImageNP(self._graphicsSceneWidget)

    # end of  def __init__(self, pParent=None)
    # --------------------------------------------------------


    # --------------------------------------------------------
    def setSimpleProgressBarPanel(self, pSimpleProcessBar=None):
    # --------------------------------------------------------
        if isinstance( pSimpleProcessBar, QtGui.QWidget ) == True:
            self.__theSimpleWaitProgressBar = pSimpleProcessBar
            self.cdImageNP.setSimpleProgressBarPanel(self.__theSimpleWaitProgressBar)
        else:
            self.__theSimpleWaitProgressBar = None
    # end of   def setSimpleProgressBarPanel()
    # --------------------------------------------------------



    # --------------------------------------------------------
    def setProgressBarWithImagePanel(self, pProcessBarWithImage=None):
    # --------------------------------------------------------
        if isinstance( pProcessBarWithImage, QtGui.QWidget ) == True:
            self.__theWaitProgressBarWithImage = pProcessBarWithImage
            self.cdImageNP.setProgressBarWithImagePanel(self.__theWaitProgressBarWithImage)

### placing the __theSimpleWaitProgressBar here instead of the pProcessBarWithImage
###  doesn't fix the reshaping/resizing problem


#             self.cdImageNP.setProgressBarWithImagePanel(pProcessBarWithImage)
        else:
            self.__theWaitProgressBarWithImage = None
    # end of   def setProgressBarWithImagePanel()
    # --------------------------------------------------------



    # --------------------------------------------------------
    def showImageInProgressBar(self, pWait=1):
    # --------------------------------------------------------
        self.__theWaitProgressBarWithImage.setTitleTextRange("Importing image sequence."," ",0, lNumberOfFilesInDir)
        
    
    
    
    
    
    # end of   showImageInProgressBar(self, pWait=1)
    # --------------------------------------------------------




    # ------------------------------------------------------------------
    # 2011 - Mitja: setImageScaleFactor() is to set the image display scale factor:
    # ------------------------------------------------------------------   
    def setImageScaleFactor(self, pValue):
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.setImageScaleFactor( pValue == "+str(pValue)+") starting." , CDConstants.DebugExcessive )
        self.theImageScaleFactor = pValue
        self.setToProcessedImage()


    # ------------------------------------------------------------------
    # 2011 - Mitja: setImageOpacity() is to set input image opacity:
    # ------------------------------------------------------------------   
    def setImageOpacity(self, pValue):
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.setImageOpacity( pValue == "+str(pValue)+") starting." , CDConstants.DebugExcessive )
        # the class global keeping track of the required opacity:
        #    0.0 = minimum = the image is completely transparent (invisible)
        #    1.0 = minimum = the image is completely opaque
        self.imageOpacity = 0.01 * (pValue)
        # 2011 - Mitja: update the CDImageLayer's parent widget,
        #   i.e. paintEvent() will be invoked regardless of the picking mode:
        # but we don't call update() here, we call it where opacity is changed
        # self._graphicsSceneWidget.scene.update()


    # ------------------------------------------------------------------
    # 2011 - Mitja: setImageOpacity() is to set :
    # ------------------------------------------------------------------   
    def setFuzzyPickTreshold(self, pValue):
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.setFuzzyPickTreshold( pValue == "+str(pValue)+") starting." , CDConstants.DebugExcessive )
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
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.setImageLoadedFromFile( pTrueOrFalse == "+str(pTrueOrFalse)+") starting." , CDConstants.DebugExcessive )
        self.imageLoadedFromFile = pTrueOrFalse
        if isinstance( self._graphicsSceneWidget, QtGui.QWidget ) == True:
            self._graphicsSceneWidget.scene.update()


    # ------------------------------------------------------------------
    # 2011 - Mitja: the setMouseTracking() function is to mimic a QLabel,
    #   which being a QWidget supports setMouseTracking(). Instead, we
    #   pass it upstream to the parent widget's QGraphicsScene:
    # ------------------------------------------------------------------   
    def setMouseTracking(self, pTrueOrFalse):
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.setMouseTracking( pTrueOrFalse == "+str(pTrueOrFalse)+") starting." , CDConstants.DebugExcessive )
        if isinstance( self._graphicsSceneWidget, QtGui.QWidget ) == True:
#             print "self._graphicsSceneWidget.view.hasMouseTracking() ==============", \
#               self._graphicsSceneWidget.view.hasMouseTracking()
#             print "                                    pTrueOrFalse ==============", \
#               pTrueOrFalse
            self._graphicsSceneWidget.view.setMouseTracking(pTrueOrFalse)
#             print "self._graphicsSceneWidget.view.hasMouseTracking() ==============", \
#               self._graphicsSceneWidget.view.hasMouseTracking()



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
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.setWidthOfFixedRaster( pGridWidth == "+str(pGridWidth)+") starting." , CDConstants.DebugExcessive )
        self.fixedRasterWidth = pGridWidth


    # ------------------------------------------------------------------
    # 2011 - Mitja: the setTheImage() function is to assign the starting QImage:
    # ------------------------------------------------------------------   
    def setTheImage(self, pImage, pImageFileName=" "):
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.setTheImage( pImage == "+str(pImage)+", pImageFileName == "+str(pImageFileName)+") starting." , CDConstants.DebugExcessive )
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

                lPainterEnded = lPainter.end()

                # copy the QImage passed as parameter into two separate instances:
                self.theImage = QtGui.QImage(lResultPixmap.toImage())
                # instead of setting manually, we call:     # self.processedImage = QtGui.QImage(lResultPixmap.toImage())
                self.setToProcessedImage()

                # also, set the input image for the NumPy image processing routines:
# 2013 TODO: restore setNPImage:
#                 self.cdImageNP.setNPLoadedFromFiles(True)
#                 self.cdImageNP.setNPImage( QtGui.QImage(lResultPixmap.toImage()), True, pImageFileName )


            else:
                # directly copy the QImage passed as parameter into two separate instances:
                self.theImage = QtGui.QImage(pImage)
                # instead of setting manually, we call:    # self.processedImage = QtGui.QImage(pImage)
                self.setToProcessedImage()

                # also, set the input image for the NumPy image processing routines:
#                 self.cdImageNP.setNPLoadedFromFiles(False)
#                 self.cdImageNP.setNPImage( QtGui.QImage(pImage), False )
# 2013 TODO: restore setNPImage:
#                 self.cdImageNP.setNPLoadedFromFiles(True)
#                 self.cdImageNP.setNPImage( QtGui.QImage(lResultPixmap.toImage()), True, pImageFileName )

            self.width = self.processedImage.width()
            self.height = self.processedImage.height()


        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.setTheImage( pImage == "+str(pImage)+", pImageFileName == "+str(pImageFileName)+") DONE." , CDConstants.DebugExcessive )
    # end of   def setTheImage(self, pImage, pImageFileName=" ")
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 2011 - Mitja: the setToProcessedImage() function restores the original
    #   QImage loaded from a file (i.e. self.theImage) back into the
    #   processed QImage (i.e. self.processedImage) undoing all color processing.
    # ------------------------------------------------------------------   
    def setToProcessedImage(self):
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.setToProcessedImage() starting." , CDConstants.DebugExcessive )
        if isinstance(self.theImage, QtGui.QImage):
            # copy self.theImage into a separate instance for processing:
            if (self.theImageScaleFactor >= 1.001) or (self.theImageScaleFactor <= 0.999) :
                self.processedImage = QtGui.QImage(self.theImage).scaled( \
                    int( float(self.processedImage.width()) * self.theImageScaleFactor ), \
                    int( float(self.processedImage.height()) * self.theImageScaleFactor ), \
                    aspectRatioMode = QtCore.Qt.KeepAspectRatio, \
                    transformMode = QtCore.Qt.SmoothTransformation  )
            else:
                self.processedImage = QtGui.QImage(self.theImage)

        else:
            self.processedImage = QtGui.QImage()
            CDConstants.printOut( "CDImageLayer.setToProcessedImage() ERROR: there is no self.theImage !!!!" , CDConstants.DebugImportant )

    # ------------------------------------------------------------
    # 2011 - Mitja: provide color-to-color distance calculation:
    # ------------------------------------------------------------
    def colorToColorIsCloseDistance(self, pC1, pC2, pDist):
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.colorToColorIsCloseDistance( pC1, pC2, pDist == "+str(pC1)+" "+str(pC2)+" "+str(pDist)+" ) starting." , CDConstants.DebugExcessive )
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
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.colorToColorIsSame( pC1, pC2 == "+str(pC1)+" "+str(pC2)+" ) starting." , CDConstants.DebugExcessive )
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
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.processImageForCloseColors( pX, pY == "+str(pX)+" "+str(pY)+" ) starting." , CDConstants.DebugExcessive )
        # create a copy of the image:
        self.processedImage = QtGui.QImage(self.theImage)

        lWidth = self.processedImage.width()
        lHeight = self.processedImage.height()
        lSeedColor = self.theImage.pixel(pX, pY)
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer: processImageForCloseColors():        01" , CDConstants.DebugTODO )
        # 2010 - Mitja: python's xrange function is more appropriate for large loops
        #   since it generates integers (the range function generates lists instead)
        for i in xrange(0, lWidth, 1):
            for j in xrange(0, lHeight, 1):
                lTmpColor = self.theImage.pixel(i,j)
                if self.colorToColorIsCloseDistance(lSeedColor, lTmpColor, self.fuzzyPickTreshold) :
                    lTmpColor = lSeedColor
                    self.processedImage.setPixel(i,j,lTmpColor)

#         self.processedPixmap = QtGui.QPixmap(self.processedImage)
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer: processImageForCloseColors():        10" , CDConstants.DebugTODO )
        # self.theImage = QtGui.QImage(self.processedImage)


    # ------------------------------------------------------------
    # 2011 - Mitja: compute the image with fuzzy pick of all colors that
    #   are "close" to the one picked, the so-called "magic wand"
    # ------------------------------------------------------------
    def processImageForFuzzyPick(self, pX, pY):
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.processImageForFuzzyPick( pX, pY == "+str(pX)+" "+str(pY)+" ) starting." , CDConstants.DebugExcessive )
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
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer: processImageForFuzzyPick():        02" , CDConstants.DebugTODO )
        self.floodFillFuzzyPick(pX, pY, lSeedColor)
#         self.processedPixmap = QtGui.QPixmap(self.processedImage)
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer: processImageForFuzzyPick():        20" , CDConstants.DebugTODO )
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
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.floodFillFuzzyPick( pX, pY, pReplacementColor == "+str(pX)+" "+str(pY)+" "+str(pReplacementColor)+" ) starting." , CDConstants.DebugExcessive )
        lWidth = self.processedImage.width()
        lHeight = self.processedImage.height()
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer: floodFillFuzzyPick():        Flood fill on a region of non-BORDER_COLOR pixels." , CDConstants.DebugTODO )
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
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.paintTheImageLayer( pThePainter == "+str(pThePainter)+" ) starting." , CDConstants.DebugExcessive )

        CDConstants.printOut("___ - DEBUG ----- CDImageLayer.paintTheImageLayer() :   [F] hello, I'm "+str(debugWhoIsTheRunningFunction())+", parent is "+str(debugWhoIsTheParentFunction())+ \
            " ||||| self.repaintEventsCounter=="+str(self.repaintEventsCounter)+ \
            " ||||| CDImageLayer.paintTheImageLayer(pThePainter=="+str(pThePainter)+")", CDConstants.DebugTODO )

        # paint into the passed QPainter parameter:
        lPainter = pThePainter

        # the QPainter has to be passed with begin() already called on it:
        # lPainter.begin()


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

        # end of if isinstance( self.processedImage, QtGui.QImage ) == True

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
        #    3 = Extract Cells = CDConstants.ImageModeExtractCells


        if (self.inputImagePickingMode == CDConstants.ImageModeExtractCells):
            # this is extract cells mode:
            # paint the content of NumPy image processing routines, if computed:
            self.cdImageNP.paintTheImageNPContent( lPainter )
            pass

        elif (self.inputImagePickingMode == CDConstants.ImageModePickColor):
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

        # end of         if (self.inputImagePickingMode == CDConstants.ImageModeExtractCells)
        # ------


        self.drawGrid(lPainter)
       
        # pop the QPainter's saved state off the stack:
        lPainter.restore()

    # end of    def paintTheImageLayer(self, pThePainter)
    # ------------------------------------------------------------------
    



    # ------------------------------------------------------------------
    # 2011 - Mitja: this function is NOT to be called directly: it is the callback handler
    #   for update() and paint() events, and it paints into the passed QPainter parameter
    # ------------------------------------------------------------------   
    def paintEvent(self, pThePainter):
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.paintEvent( pThePainter == "+str(pThePainter)+" ) starting." , CDConstants.DebugExcessive )
   
        # one paint cycle has been called:
        self.repaintEventsCounter = self.repaintEventsCounter + 1

        CDConstants.printOut("___ - DEBUG ----- CDImageLayer.paintEvent() :   [G] hello, I'm "+str(debugWhoIsTheRunningFunction())+", parent is "+str(debugWhoIsTheParentFunction())+ \
            " ||||| self.repaintEventsCounter=="+str(self.repaintEventsCounter)+ \
            " ||||| CDImageLayer.paintEvent(pThePainter=="+str(pThePainter)+")", CDConstants.DebugTODO )

        # 2011 - Mitja: call our function doing the actual drawing,
        #   passing along the QPainter parameter received by paintEvent():
        self.paintTheImageLayer(pThePainter)


    # ------------------------------------------------------------------
    def drawGrid(self, pThePainter):
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.drawGrid( pThePainter == "+str(pThePainter)+" ) starting." , CDConstants.DebugExcessive )
   
        # here there would be a grid, but we don't want it:
        # draw vertical lines:
#         for x in xrange(0, self.pifSceneWidth, self.fixedRasterWidth):
#             #draw.line([(x, 0), (x, h)], width=2, fill='#000000')
#             pThePainter.setPen(QtGui.QColor(QtCore.Qt.green))
#             pThePainter.drawLine(x, 0, x, self.pifSceneHeight)
#         # draw horizontal lines:
#         for y in xrange(0, self.pifSceneHeight, self.fixedRasterWidth):
#             #draw.line([(0, y), (w, y)], width=2, fill='#000000')
#             pThePainter.setPen(QtGui.QColor(QtCore.Qt.blue))
#             pThePainter.drawLine(0, y, self.pifSceneWidth, y)

        # draw boundary frame lines:

        # this would a solid outline line, we don't want it:
        # pThePainter.setPen(QtGui.QColor(QtCore.Qt.red))

        # draw the rectangular outline in two colors, solid and dotted:

        lOutlineColor = QtGui.QColor(35, 166, 94)
        lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        lOutlinePen.setCosmetic(True) # cosmetic pen = width always 1 pixel wide, independent of pThePainter's transformation set
        pThePainter.setPen(lOutlinePen)

        pThePainter.drawLine(0, 0, 0, self.pifSceneHeight-1)
        pThePainter.drawLine(self.pifSceneWidth-1, 0, self.pifSceneWidth-1, self.pifSceneHeight-1)
        pThePainter.drawLine(0, 0, self.pifSceneWidth-1, 0)
        pThePainter.drawLine(0, self.pifSceneHeight-1, self.pifSceneWidth-1, self.pifSceneHeight-1)

        lOutlineColor = QtGui.QColor(219, 230, 249)
        lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.DotLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        lOutlinePen.setCosmetic(True) # cosmetic pen = width always 1 pixel wide, independent of pThePainter's transformation set
        pThePainter.setPen(lOutlinePen)
        # vertical lines:       
        pThePainter.drawLine(0, 0, 0, self.pifSceneHeight-1)
        pThePainter.drawLine(self.pifSceneWidth-1, 0, self.pifSceneWidth-1, self.pifSceneHeight-1)

        lOutlineColor = QtGui.QColor(255, 0, 0)
        lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.DotLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        lOutlinePen.setCosmetic(True) # cosmetic pen = width always 1 pixel wide, independent of pThePainter's transformation set
        pThePainter.setPen(lOutlinePen)
        # bottom (y=0) line
        pThePainter.drawLine(0, 0, self.pifSceneWidth-1, 0)

        lOutlineColor = QtGui.QColor(0, 255, 255)
        lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.DotLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        lOutlinePen.setCosmetic(True) # cosmetic pen = width always 1 pixel wide, independent of pThePainter's transformation set
        pThePainter.setPen(lOutlinePen)
        # top (y=0) line
        pThePainter.drawLine(0, self.pifSceneHeight-1, self.pifSceneWidth-1, self.pifSceneHeight-1)

#         print "2010 DEBUG:", self.repaintEventsCounter, "CDImageLayer.drawGrid() DONE - WWIIDDTTHH(image,pifScene) =", self.width,self.pifSceneWidth, "HHEEIIGGHHTT(image,pifScene) =", self.height,self.pifSceneHeight
       





    # ---------------------------------------------------------
    # the reject() function is called when the user presses the <ESC> keyboard key.
    #   We override the default and implicitly include the equivalent action of
    #   <esc> being the same as clicking the "Cancel" button, as in well-respected GUI paradigms:
    # ---------------------------------------------------------
    def reject(self):
        super(CDImageLayer, self).reject()
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer:    reject() DONE" , CDConstants.DebugTODO )






# FIX ESC KEY in IMAGE DRAWING!!!



    # ------------------------------------------------------------------
    def mousePressEvent(self, pEvent):       
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.mousePressEvent( pEvent == "+str(pEvent)+" ) starting." , CDConstants.DebugExcessive )
        # 2010 - Mitja: track click events of the mouse left button only:
        if pEvent.button() == QtCore.Qt.LeftButton:

            lX = pEvent.scenePos().x()
            lY = pEvent.scenePos().y()

            # 2010 - Mitja: add freeform shape drawing on the top of image,
            #    where the global for keeping track of what's been changed can be:
            #    0 = Color Pick = CDConstants.ImageModePickColor
            #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
            #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon
            #    3 = Extract Cells = CDConstants.ImageModeExtractCells
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
                   
            elif (self.inputImagePickingMode == CDConstants.ImageModeExtractCells):
                # this is extract cells mode:
                color = self.theImage.pixel(lX, lY)
                self.myMouseX = lX
                self.myMouseY = lY
                # 2011 - Mitja: to pick a color region in the image for the PIFF scene,
                #   uncomment only ONE of these two methods - either fuzzy pick or close colors:

                self.cdImageNP.theTrueExtractCells()
#                 self.cdImageNP.theTrueExtractCells(self.myMouseX, self.myMouseY)




                self.emit(QtCore.SIGNAL("mousePressedInImageLayerSignal()"))
                   
            # end of    if (self.inputImagePickingMode == CDConstants.ImageModePickColor)
            # ----


            # 2010 - Mitja: update the CDImageLayer's parent widget,
            #   i.e. paintEvent() will be invoked regardless of the picking mode:
            self._graphicsSceneWidget.scene.update()



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
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.mouseReleaseEvent( pEvent == "+str(pEvent)+" ) starting." , CDConstants.DebugExcessive )

        lX = pEvent.scenePos().x()
        lY = pEvent.scenePos().y()

        # 2010 - Mitja: add freeform shape drawing on the top of image,
        #    where the global for keeping track of what's been changed can be:
        #    0 = Color Pick = CDConstants.ImageModePickColor
        #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
        #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon
        #    3 = Extract Cells = CDConstants.ImageModeExtractCells
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
                    self._graphicsSceneWidget.scene.mousePressEvent( \
                        QtGui.QMouseEvent( QtCore.QEvent.GraphicsSceneMousePress, \
                                           QtCore.QPoint(0,0), \
                                           QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier), \
                        theGraphicsPathItem )
                    self.theCurrentPath = []
                    self.myPathHighlight = False
   
                # 2010 - Mitja: update the CDImageLayer's parent widget,
                #   i.e. paintEvent() will be invoked:
                self._graphicsSceneWidget.scene.update()

        elif (self.inputImagePickingMode == CDConstants.ImageModeDrawPolygon):
        # this is polygon drawing mode:
            # 2010 - Mitja: track release events of the mouse left button only:
            if (pEvent.button() == QtCore.Qt.LeftButton):
                self.myMouseLeftDown = False
                self._graphicsSceneWidget.scene.update()

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
                    self._graphicsSceneWidget.scene.mousePressEvent( \
                        QtGui.QMouseEvent( QtCore.QEvent.GraphicsSceneMousePress, \
                                           QtCore.QPoint(0,0), \
                                           QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier), \
                        theGraphicsPathItem )
                    self.theCurrentPath = []
                    self.myPathHighlight = False


        elif (self.inputImagePickingMode == CDConstants.ImageModeExtractCells):
        # this is extract cells mode:
            pass

        # end of  if (self.inputImagePickingMode == CDConstants.ImageModePickColor)
        # ----


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

        CDConstants.printOut( "___ - DEBUG ----- 2012 DEBUG: CDImageLayer.mouseReleaseEvent() - pEvent(x,y)=("+str(lX)+","+str(lY)+") : done" , CDConstants.DebugTODO )

    # end of def mouseReleaseEvent(self, pEvent)
    # ---------------------------------------------------------



    # ---------------------------------------------------------
    # 2012 - Mitja - keyReleaseEvent() "fake" event handler
    #   added to handle pressing the <esc> key.
    #   This function is called by DiagramScene's real keyReleaseEvent handler.
    # ---------------------------------------------------------
    def keyReleaseEvent(self, pEvent):
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.keyReleaseEvent( pEvent == "+str(pEvent)+" ) starting." , CDConstants.DebugExcessive )

        if (pEvent.key() == QtCore.Qt.Key_Escape):
            #    0 = Color Pick = CDConstants.ImageModePickColor
            #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
            #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon
            #    3 = Extract Cells = CDConstants.ImageModeExtractCells
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
        self._graphicsSceneWidget.scene.update()

    # end of   def keyReleaseEvent(self, pEvent)
    # ---------------------------------------------------------



    # ---------------------------------------------------------
    def mouseMoveEvent(self, pEvent):
        CDConstants.printOut( "___ - DEBUG ----- CDImageLayer.mouseMoveEvent( pEvent == "+str(pEvent)+" ) starting." , CDConstants.DebugExcessive )
    
        # 2010 - Mitja: add freeform shape drawing on the top of image,
        #    where the global for keeping track of what's been changed can be:
        #    0 = Color Pick = CDConstants.ImageModePickColor
        #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
        #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon
        #    3 = Extract Cells = CDConstants.ImageModeExtractCells

        lX = pEvent.scenePos().x()
        lY = pEvent.scenePos().y()
        lColorAtMousePos = self.theImage.pixel(lX, lY)
        # we pass a "dict" parameter with the signalThatMouseMoved parameter, so that
        #   both the mouse x,y coordinates as well as color information can be passed around:

        if ( self.theImage.rect().contains(lX, lY) ):
            lColorAtMousePos = self.theImage.pixel(lX, lY)
        else:
            lColorAtMousePos = QtGui.QColor(255,255,255)
        lR = int( QtGui.QColor(lColorAtMousePos).red() )
        lG = int( QtGui.QColor(lColorAtMousePos).green() )
        lB = int( QtGui.QColor(lColorAtMousePos).blue() )
        lDict = { \
            0: str(int(lX)), \
            1: str(int(lY)), \
            #  the depth() function is not part of QGraphicsScene, we could add it for completeness?
            2: lR , \
            3: lG , \
            4: lB 
            }


        if (self.inputImagePickingMode == CDConstants.ImageModePickColor):
        # this is color-picking mode:
            if (self.myMouseLeftDown == False) and \
              (self._graphicsSceneWidget.view.hasMouseTracking() == True):
                # print lX, lY
                # 2011 - Mitja: to have real-time visual feedback while moving the mouse,
                #   without actually picking a color region in the image for the PIFF scene,
                #   uncomment one of these two methods:
                # self.processImageForFuzzyPick(lX, lY)
                # self.processImageForCloseColors(lX, lY)
                # self._graphicsSceneWidget.scene.update()
                pass
        elif (self.inputImagePickingMode == CDConstants.ImageModeDrawFreehand):
        # this is freeform drawing mode:
            if (self.myMouseLeftDown == True) and \
              (self._graphicsSceneWidget.view.hasMouseTracking() == False):
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
                self._graphicsSceneWidget.scene.update()
        elif (self.inputImagePickingMode == CDConstants.ImageModeDrawPolygon):
        # this is polygon drawing mode:
            if (self.myMouseLeftDown == False) and \
              (self._graphicsSceneWidget.view.hasMouseTracking() == True):
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
                self._graphicsSceneWidget.scene.update()

        if (self.inputImagePickingMode == CDConstants.ImageModeExtractCells):
        # this is extract cells mode:
            if (self.myMouseLeftDown == False) and \
              (self._graphicsSceneWidget.view.hasMouseTracking() == True):
                # print lX, lY
                # 2012 - Mitja: to have real-time visual feedback while moving the mouse,
                #   without actually generating color regions from the image to the PIFF scene,
                #   uncomment this method:
                self.cdImageNP.imageNPpreview(lX, lY, lR, lG, lB)
                self._graphicsSceneWidget.scene.update()
                pass

        # finally, signal that the mouse moved:
        self.signalThatMouseMoved.emit(lDict)


# end of  class CDImageLayer(QtCore.QObject)
# ======================================================================





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
