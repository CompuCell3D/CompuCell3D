#!/usr/bin/env python

from PyQt4 import QtCore, QtGui


import PyQt4.QtGui
import PyQt4.QtCore

import math   # we need to import math to use sqrt() and such
import sys    # for get/setrecursionlimit() and sys.version_info

# 2012 - Mitja: advanced debugging tools,
#   work only on Posix-compliant systems so far (no MS-Windows)
#   commented out for now:
# import os     # for kill() and getpid()
# import signal # for signal()

from timeit import Timer    #    for timing testing purposes

import time    # for sleep()

import random  # for semi-random-colors

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

# 2010 - Mitja: simple external class for drawing a progress bar widget:
from cdWaitProgressBar import CDWaitProgressBar

# 2011 - Mitja: simple external class for drawing a progress bar widget + an image:
from cdWaitProgressBarWithImage import CDWaitProgressBarWithImage



# ----------------------------------------------------------------------
# 2011- Mitja: additional layer to show a sequence of images for CellDraw
#   on the top of the QGraphicsScene-based cell/region editor
# ------------------------------------------------------------
# This class draws an image from an array of pixels loaded from a sequence of files
# ------------------------------------------------------------
class CDImageSequence(QtCore.QObject):

    # 2011 - Mitja: a signal for image sequence resizing. Has to be handled in CDDiagramScene!
    signalThatImageSequenceResized = QtCore.pyqtSignal(dict)

    # 2011 - Mitja: a signal for the setting of self.theCurrentIndex. Has to be handled in CDDiagramScene!
    signalThatCurrentIndexSet = QtCore.pyqtSignal(dict)

    # --------------------------------------------------------
    def __init__(self, pParent=None):

        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: __init__( pParent == "+str(pParent)+") " , CDConstants.DebugExcessive )

        QtCore.QObject.__init__(self, pParent)

        # a flag to show if CDImageSequence holds a sequence of images
        #    as loaded from files:
        self.imageSequenceLoaded = False

        # a string containing the full path to the directory holding the image files:
        self.imageSequencePathString = ""

        # the class global keeping track of the selected image within the sequence:
        #    0 = minimum = the first image in the sequence stack
        self.theCurrentIndex = 0

        # the class global keeping track of the selected color/type for the sequence:
        self.theImageSequenceColor = QtGui.QColor(QtCore.Qt.magenta)

        # the class global keeping track of the "wall" color/type for the sequence:
        self.theImageSequenceWallColor = QtGui.QColor(QtCore.Qt.green)


        # the class global keeping track of the bit-flag modes for generating PIFF from displayed imported image sequence:
        #    0 = Use Discretized Images to B/W = CDConstants.ImageSequenceUseDiscretizedToBWMode
        #    1 = Region 2D Edge = CDConstants.ImageSequenceUseEdge
        #    2 = Region 3D Contours = CDConstants.ImageSequenceUse3DContours
        #    3 = Region 3D Volume = CDConstants.ImageSequenceUse3DVolume
        #    4 = Region Cell Seeds = CDConstants.ImageSequenceUseAreaSeeds
        self.theProcessingModeForImageSequenceToPIFF = (1 << CDConstants.ImageSequenceUseAreaSeeds)

        # bin() does not exist in Python 2.5:
        if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: __init__() bin(self.theProcessingModeForImageSequenceToPIFF) == "+str(bin(self.theProcessingModeForImageSequenceToPIFF)) , CDConstants.DebugExcessive )
        else:
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: __init__() self.theProcessingModeForImageSequenceToPIFF == "+str(self.theProcessingModeForImageSequenceToPIFF) , CDConstants.DebugExcessive )



        # the size of all images (width=x, height=y) in the sequence
        #   and number of images (z) in the sequence:
        self.sizeX = 0
        self.sizeY = 0
        self.sizeZ = 0

        # 2011 - Mitja: globals storing pixel-size data for a complete image sequence
        #
        # a numpy-based array global of size 1x1x1x1,
        #     it'll be resized when the sequence of images is loaded, i.e.
        #     the actually used array will be set in resetSequenceDimensions()
        # the 4 dimensions are, from the slowest (most distant)
        #     to the fastest (closest data to each other):
        #   z = image layers, y = height, x = width, [b g r], 
        self.imageSequenceArray = numpy.zeros( (1, 1, 1, 1), dtype=numpy.uint8 )
        self.volumeSequenceArray = numpy.zeros( (1, 1, 1, 1), dtype=numpy.uint8 )
        self.edgeSequenceArray = numpy.zeros( (1, 1, 1, 1), dtype=numpy.uint8 )
        self.contoursSequenceArray = numpy.zeros( (1, 1, 1, 1), dtype=numpy.uint8 )

        # another  couple of numpy arrays to store flags for already computed data:
        # 
        self.imageInSequenceIsReadyFlags = numpy.zeros( (1), dtype=numpy.bool )
        self.edgeInSequenceIsReadyFlags = numpy.zeros( (1), dtype=numpy.bool )
        # and a flag for everything ready in the 3D contours:
        self.contoursAreReadyFlag = False

        # and a dict to store all filename strings, one per image layer
        self.imageSequenceFileNames = dict()
        self.imageSequenceFileNames[0] = " "

        self.pifSceneWidth = 120
        self.pifSceneHeight = 90
        self.pifSceneDepth = 100

        self.resetSequenceDimensions( \
            self.pifSceneWidth, \
            self.pifSceneHeight, \
            self.pifSceneDepth )

        self.fixedRasterWidth = 10

        # the "theCurrentImage" QImage is the selected image loaded from a sequence of files:
        lBoringPixMap = QtGui.QPixmap(self.pifSceneWidth, self.pifSceneHeight)
        lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.transparent) )
        self.theCurrentImage = QtGui.QImage(lBoringPixMap)

        # the "theCurrentVolumeSliceImage" QImage is the selected volume slice image discretized from the sequence:
        lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.transparent) )
        self.theCurrentVolumeSliceImage = QtGui.QImage(lBoringPixMap)

        # the "theCurrentEdge" QImage is the selected edge image detected from the sequence:
        lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.transparent) )
        self.theCurrentEdge = QtGui.QImage(lBoringPixMap)

        # the "theCurrentContour" QImage is the selected 3D contour slice as detected from the sequence:
        lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.transparent) )
        self.theCurrentContour = QtGui.QImage(lBoringPixMap)

        # the scale factor is the zoom factor for viewing the sequence of images, separate from the PIFF scene zoom:
        self.scaleFactor = 1.0

        if isinstance( pParent, QtGui.QWidget ) == True:
            self.graphicsSceneWindow = pParent
        else:
            self.graphicsSceneWindow = None

        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: __init__() graphicsSceneWindow == "+str(self.graphicsSceneWindow)+" " , CDConstants.DebugExcessive )
        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: __init__() imageInSequenceIsReadyFlags == "+str(self.imageInSequenceIsReadyFlags)+" " , CDConstants.DebugExcessive )
        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: __init__() edgeInSequenceIsReadyFlags == "+str(self.edgeInSequenceIsReadyFlags)+" " , CDConstants.DebugExcessive )
        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: __init__() contoursAreReadyFlag == "+str(self.contoursAreReadyFlag)+" " , CDConstants.DebugExcessive )

        # debugging counter, how many times has the paint function been called:
        self.repaintEventsCounter = 0



    # ------------------------------------------------------------------
    # 2011 - Mitja: setScaleZoom() is to set the image display scale/zoom factor:
    # TODO - this is at the moment never called from ANYWHERE
    # ------------------------------------------------------------------   
    def setScaleZoom(self, pValue):
        # TODO - this is at the moment never called from ANYWHERE
        self.scaleFactor = pValue
        self.setToProcessedImage()




    # ------------------------------------------------------------------
    # 2011 - Mitja: resetSequenceDimensions() is to set x,y,z for the sequence:
    # ------------------------------------------------------------------   
    def resetSequenceDimensions(self, pX, pY, pZ):

        self.sizeX = pX
        self.sizeY = pY
        self.sizeZ = pZ

        # a fresh numpy-based array - the 4 dimensions are:
        #   z = image layers,  y = height,  x = width,  [b g r]  
        #   (i.e. indexed from slowest to fastest)

        self.imageSequenceArray = numpy.zeros( \
            (self.sizeZ,  self.sizeY,  self.sizeX,  3), \
            dtype=numpy.uint8  )
        self.edgeSequenceArray = numpy.zeros( \
            (self.sizeZ,  self.sizeY,  self.sizeX,  3), \
            dtype=numpy.uint8  )
        self.contoursSequenceArray = numpy.zeros( \
            (self.sizeZ,  self.sizeY,  self.sizeX,  3), \
            dtype=numpy.uint8  )
        self.volumeSequenceArray = numpy.zeros( \
            (self.sizeZ,  self.sizeY,  self.sizeX,  3), \
            dtype=numpy.uint8  )

        # reset to fresh and empty (all False) flag arrays as well:
        self.imageInSequenceIsReadyFlags = numpy.zeros( (self.sizeZ), dtype=numpy.bool )
        self.edgeInSequenceIsReadyFlags = numpy.zeros( (self.sizeZ), dtype=numpy.bool )
        self.contoursAreReadyFlag = False

        # reset to fresh and empty (all " ") filename strings as well:
        for i in xrange(self.sizeZ):
            self.imageSequenceFileNames[i] = " "

        # emit a signal to update image sequence size GUI controls:
        if ( int(self.sizeZ) == 1 ) :
            lLabel = "image"
        else:
            lLabel = "images"

        lDict = { \
            0: str(int(self.sizeX)), \
            1: str(int(self.sizeY)), \
            2: str(int(self.sizeZ)), \
            3: str(lLabel) \
            }

        self.signalThatImageSequenceResized.emit(lDict)

        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: resetSequenceDimensions() self.sizeX,Y,Z == "+str(self.sizeX)+" "+str(self.sizeY)+" "+str(self.sizeZ)+" " , CDConstants.DebugVerbose )




        #   an empty array, into which to write pixel values,
        #   one Z layer for each image in the sequence,
        #   and we use numpy.int32 as data type to hold RGBA values:
        # self.imageSequenceArray = numpy.zeros( (self.sizeY, self.sizeX, self.sizeZ), dtype=numpy.int )


#         TODO: this testing only necessary when NO image sequence loaded:

        # if there is no image sequence loaded, then do nothing after clearing the array:



#         if  (self.imageSequenceLoaded == False):
#     
#             # set test array content:
# 
#             # show a panel containing a progress bar:    
#             lProgressBarPanel=CDWaitProgressBar("Test content image sequence array x="+str(self.sizeX)+" y="+str(self.sizeY)+"  z="+str(self.sizeZ), 100)
#             lProgressBarPanel.show()
#             lProgressBarPanel.setRange(0, self.sizeZ)
# 
# 
#             # for each image in sequence:
#             for k in xrange(0, self.sizeZ, 1):
#                 # for each i,j point in an image:
#                 for i in xrange(0, self.sizeX, 1):
#                     for j in xrange(0, self.sizeY, 1):
#                         if (i == j) and (j == k):
#                             self.imageSequenceArray[k, j, i, 0] = numpy.uint8 (127)
#                         else:
#                             self.imageSequenceArray[k, j, i, 0] = numpy.uint8 ( 0 )
# #                        print "i,j,k [",i,j,k,"] =", self.imageSequenceArray[j, i, k]
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



    # ------------------------------------------------------------------
    # 2011 - Mitja: getSequenceDimensions() is to get x,y,z for the sequence:
    # ------------------------------------------------------------------   
    def getSequenceDimensions(self):

        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: getSequenceDimensions() self.sizeX,Y,Z == "+str(self.sizeX)+" "+str(self.sizeY)+" "+str(self.sizeZ)+" " , CDConstants.DebugExcessive )

        return (self.sizeX, self.sizeY, self.sizeZ)



    # ------------------------------------------------------------------
    # 2011 - Mitja: imageCurrentImage() creates theCurrentImage, theCurrentEdge and theCurrentVolumeSliceImage
    #   from the current layer in the sequence arrays:
    # ------------------------------------------------------------------   
    def imageCurrentImage(self):

        # do nothing if the current array size isn't anything:
        if (self.sizeX <= 0) or (self.sizeY <= 0) or (self.sizeZ <= 0) :
            return

        # do nothing if the current image index doesn't have a corresponding image:
        if (self.theCurrentIndex < 0) or (self.theCurrentIndex >= self.sizeZ) :
            return

        # obtain the current image data from one layer in the image numpy array 
        lTmpOneLayerArray = self.imageSequenceArray[self.theCurrentIndex]
        self.theCurrentImage = self.rgb2qimage(lTmpOneLayerArray)

        # obtain the current volume slice data from one layer in the volume numpy array 
        lTmpOneLayerArray = self.volumeSequenceArray[self.theCurrentIndex]
        self.theCurrentVolumeSliceImage = self.rgb2qimageKtoBandA(lTmpOneLayerArray)

        # check if the current image has its edge already computed; if it doesn't, then compute it now:
        if (self.edgeInSequenceIsReadyFlags[self.theCurrentIndex] == False):
            # self.theTrueComputeCurrentEdge()
            # TODO: calling computeCurrentEdge() is to be used just for testing theTrueComputeCurrentEdge() with a timer, afterwards revert to calling theTrueComputeCurrentEdge() directly:
            self.computeCurrentEdge()
        # obtain the current edge data from one layer in the edge numpy array 
        lTmpOneLayerArray = self.edgeSequenceArray[self.theCurrentIndex]
        self.theCurrentEdge = self.rgb2qimageWtoRandA(lTmpOneLayerArray)

        # if 3D contours have to be painted, check if the 3D contours have already been computed, and if not, do so now:
        if ( self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUse3DContours) ) and \
            (self.contoursAreReadyFlag == False):
            # TODO: calling computeContours() is to be used just for testing ...() with a timer, afterwards revert to calling ...() directly:
            self.computeContours()
            
        # obtain the current contour data from one layer in the contour numpy array 
        lTmpOneLayerArray = self.contoursSequenceArray[self.theCurrentIndex]
        self.theCurrentContour = self.rgb2qimageWtoGandA(lTmpOneLayerArray)


#### TODO: fix B/W toggle so that it computes correctly for 3D contours (maybe never gets there?) and for 2D contours (delayed 1 toggle?)


#         lBackgroundRect = QtCore.QRectF( QtCore.QRect(0, 0, self.sizeX, self.sizeY) )
#         lPixmap = QtGui.QPixmap(lBackgroundRect.width(), lBackgroundRect.height())
#         lPixmap.fill( QtGui.QColor(QtCore.Qt.transparent) )
#         lPainter = QtGui.QPainter(lPixmap)
# 
#         k = self.theCurrentIndex
# 
#         lTmpRect = QtCore.QRectF( QtCore.QRect(0, 0, 1, 1) )
#         lTmpPixmap  = QtGui.QPixmap( lTmpRect.width(), lTmpRect.height() )
# 
#         lColor = QtGui.QColor()
# 
#         # for each i,j point in an image:
#         for i in xrange(0, self.sizeX, 1):
#             for j in xrange(0, self.sizeY, 1):
# 
#                 r = int ( self.imageSequenceArray[k, j, i, 0] )
#                 g = int ( self.imageSequenceArray[k, j, i, 1] )
#                 b = int ( self.imageSequenceArray[k, j, i, 2] )
#                 lColor.setRgb( r,g,b )
#                 lTmpPixmap.fill( lColor )
# 
#                 lPainter.drawPixmap(QtCore.QPoint(i,j), lTmpPixmap)
# 
#         lPainter.end()
# 
#         self.theCurrentImage = lPixmap.toImage()

        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: imageCurrentImage() self.theCurrentIndex == "+str(self.theCurrentIndex)+ " DONE." , CDConstants.DebugVerbose )



    # ------------------------------------------------------------------
    # 2011 - Mitja: normalizeAllImages() images theImage from the current layer in the sequence:
    #    
    # DO NOT USE: the following suggestion thrashes the system!!!!... 2GB of VM dump on disk and stuck:
    #     
    #         # convert to floats in the range [0.0, 1.0] :
    #         f = (self.imageSequenceArray - self.imageSequenceArray.min()) \
    #             / float(self.imageSequenceArray.max() - self.imageSequenceArray.min())
    #     
    #         # convert back to bytes:
    #         self.imageSequenceArray = (f * 255).astype(numpy.uint8)
    # 
    # ------------------------------------------------------------------   
    def normalizeAllImages(self):

        # do nothing if the current array size isn't anything:
        if (self.sizeX <= 0) or (self.sizeY <= 0) or (self.sizeZ <= 0) :
            return

        # do nothing if the current image index doesn't have a corresponding image:
        if (self.theCurrentIndex < 0) or (self.theCurrentIndex >= self.sizeZ) :
            return


        lTheMaxImageSequenceValue = self.imageSequenceArray.max()
        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: normalizeAllImages() lTheMaxImageSequenceValue == "+str(lTheMaxImageSequenceValue) , CDConstants.DebugVerbose )


        if (lTheMaxImageSequenceValue >= 1) and (lTheMaxImageSequenceValue < 128):

            if (lTheMaxImageSequenceValue < 2) :
                lTmpArray = numpy.left_shift( self.imageSequenceArray, 7)
                CDConstants.printOut( " imageSequenceArray 7 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxImageSequenceValue < 4) :
                lTmpArray = numpy.left_shift( self.imageSequenceArray, 6)
                CDConstants.printOut( " imageSequenceArray 6 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxImageSequenceValue < 8) :
                lTmpArray = numpy.left_shift( self.imageSequenceArray, 5)
                CDConstants.printOut( " imageSequenceArray 5 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxImageSequenceValue < 16) :
                lTmpArray = numpy.left_shift( self.imageSequenceArray, 4)
                CDConstants.printOut( " imageSequenceArray 4 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxImageSequenceValue < 32) :
                lTmpArray = numpy.left_shift( self.imageSequenceArray, 3)
                CDConstants.printOut( " imageSequenceArray 3 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxImageSequenceValue < 64) :
                lTmpArray = numpy.left_shift( self.imageSequenceArray, 2)
                CDConstants.printOut( " imageSequenceArray 2 bit shift done " , CDConstants.DebugExcessive )
            else :   # i.e. (lTheMaxImageSequenceValue < 128) :
                lTmpArray = numpy.left_shift( self.imageSequenceArray, 1)
                CDConstants.printOut( " imageSequenceArray 1 bit shift done " , CDConstants.DebugExcessive )

            self.imageSequenceArray = lTmpArray

            lTheMaxImageSequenceValue = self.imageSequenceArray.max()
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: normalizeAllImages() now new lTheMaxImageSequenceValue == "+str(lTheMaxImageSequenceValue) , CDConstants.DebugVerbose )

        # end of  if (lTheMaxImageSequenceValue >= 1) and (lTheMaxImageSequenceValue < 128)



        # discretizing to black/white, i.e. the "volume array" values are to be first set to either 0 or 1:
        lTmpZerosLikeArray = numpy.zeros_like( self.imageSequenceArray )
        lTmpBoolArray = numpy.not_equal (self.imageSequenceArray , lTmpZerosLikeArray)
        self.volumeSequenceArray = lTmpBoolArray.astype (numpy.uint8)

        lTheMaxVolumeArrayValue = self.volumeSequenceArray.max()
        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: normalizeAllImages() lTheMaxVolumeArrayValue == "+str(lTheMaxVolumeArrayValue) , CDConstants.DebugVerbose )


        if (lTheMaxVolumeArrayValue >= 1) and (lTheMaxVolumeArrayValue < 128):

            if (lTheMaxVolumeArrayValue < 2) :
                lTmpArray = numpy.left_shift( self.volumeSequenceArray, 7)
                CDConstants.printOut( " volumeSequenceArray 7 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxVolumeArrayValue < 4) :
                lTmpArray = numpy.left_shift( self.volumeSequenceArray, 6)
                CDConstants.printOut( " volumeSequenceArray 6 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxVolumeArrayValue < 8) :
                lTmpArray = numpy.left_shift( self.volumeSequenceArray, 5)
                CDConstants.printOut( " volumeSequenceArray 5 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxVolumeArrayValue < 16) :
                lTmpArray = numpy.left_shift( self.volumeSequenceArray, 4)
                CDConstants.printOut( " volumeSequenceArray 4 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxVolumeArrayValue < 32) :
                lTmpArray = numpy.left_shift( self.volumeSequenceArray, 3)
                CDConstants.printOut( " volumeSequenceArray 3 bit shift done " , CDConstants.DebugExcessive )
            elif (lTheMaxVolumeArrayValue < 64) :
                lTmpArray = numpy.left_shift( self.volumeSequenceArray, 2)
                CDConstants.printOut( " volumeSequenceArray 2 bit shift done " , CDConstants.DebugExcessive )
            else :   # i.e. (lTheMaxVolumeArrayValue < 128) :
                lTmpArray = numpy.left_shift( self.volumeSequenceArray, 1)
                CDConstants.printOut( " volumeSequenceArray 1 bit shift done " , CDConstants.DebugExcessive )

            self.volumeSequenceArray = lTmpArray

            lTheMaxVolumeArrayValue = self.volumeSequenceArray.max()
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: normalizeAllImages() now new lTheMaxVolumeArrayValue == "+str(lTheMaxVolumeArrayValue) , CDConstants.DebugVerbose )

        # end of  if (lTheMaxVolumeArrayValue >= 1) and (lTheMaxVolumeArrayValue < 128)



# 
# 
#         # show a panel containing a progress bar:    
#         lProgressBarPanel=CDWaitProgressBar("Test normalizeAllImages array x="+str(self.sizeX)+" y="+str(self.sizeY)+"  z="+str(self.sizeZ), 100)
#         lProgressBarPanel.show()
#         lProgressBarPanel.setRange(0, self.sizeZ)
# 
# 
#         # for each image in sequence:
#         for k in xrange(0, self.sizeZ, 1):
#             # for each i,j point in an image:
#             for i in xrange(0, self.sizeX, 1):
#                 for j in xrange(0, self.sizeY, 1):
#                     if  (lTmpBoolArray[k, j, i, 0] == True) or (lTmpBoolArray[k, j, i, 1] == True) or (lTmpBoolArray[k, j, i, 2] == True) :
#                         CDConstants.printOut(  "at i,j,k ["+str(i)+","+str(j)+","+str(k)+"] self.imageSequenceArray[k,j,i] = "+str(self.imageSequenceArray[k, j, i])+\
#                             "  lTmpBoolArray[k,j,i] = "+str(lTmpBoolArray[k, j, i])+"  self.volumeSequenceArray[k,j,i] = "+str(self.volumeSequenceArray[k,j,i]), CDConstants.DebugExcessive )
# 
# #                     if  (self.imageSequenceArray[k, j, i, 0] > 0) or (self.imageSequenceArray[k, j, i, 1] > 0) or (self.imageSequenceArray[k, j, i, 2] > 0) :
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
# #                        print "i,j,k [",i,j,k,"] =", self.imageSequenceArray[j, i, k]
# #                     print "-----------------------------------"
# #                 print "===================================="
#                     
#             lProgressBarPanel.setValue(k)
# 
#         # close the panel containing a progress bar:
#         lProgressBarPanel.maxProgressBar()
#         lProgressBarPanel.accept()




#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "type(self.imageSequenceArray) = ", type(self.imageSequenceArray)
#         print "type(self.imageSequenceArray[0,0,0,0]) = ", type(self.imageSequenceArray[0,0,0,0]), " and contains ", self.imageSequenceArray[0,0,0,0]
#         print "self.imageSequenceArray.max() = ", self.imageSequenceArray.max()
#         print "self.imageSequenceArray.min() = ", self.imageSequenceArray.min()
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
#         print "type(self.volumeSequenceArray) = ", type(self.volumeSequenceArray)
#         print "type(self.volumeSequenceArray[0,0,0,0]) = ", type(self.volumeSequenceArray[0,0,0,0]), " and contains ", self.volumeSequenceArray[0,0,0,0]
#         print "self.volumeSequenceArray.max() = ", self.volumeSequenceArray.max()
#         print "self.volumeSequenceArray.min() = ", self.volumeSequenceArray.min()
#         # print self.volumeSequenceArray
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"
#         print "--------------------------------------------------------------------------------"


    # end of     def normalizeAllImages(self)
    # ------------------------------------------------------------------   
    # ------------------------------------------------------------------   






# 
# 
#     # ------------------------------------------------------------------
#     # 2011 - Mitja: theTrueSetCurrentImage() sets pixel values in one layer from an image in the sequence:
#     # ------------------------------------------------------------------   
#     def theTrueSetCurrentImage(self):
# 
# #         lWidth = self.theCurrentImage.width()
# #         lHeight = self.theCurrentImage.height()
# 
#         lTmpArrayBGRA = self.qimage2numpy(self.theCurrentImage, "array")
#         self.imageSequenceArray[self.theCurrentIndex] = lTmpArrayBGRA



#         print lTmpArrayBGRA.ndim     -> 3
#         print lTmpArrayBGRA.size     -> 517608
#         print lTmpArrayBGRA.itemsize -> 1
#         print lTmpArrayBGRA.nbytes   -> 
#         print lTmpArrayBGRA.dtype    -> uint8
#         print lTmpArrayBGRA.flags    ->   C_CONTIGUOUS : False
#                                           F_CONTIGUOUS : False
#                                           OWNDATA : False
#                                           WRITEABLE : False
#                                           ALIGNED : True
#                                           UPDATEIFCOPY : False






#         for i in xrange(0, lWidth, 1):
#             for j in xrange(0, lHeight, 1):
# 
#                 lRGBAColorAtPixel = lTmpArrayBGRA[j,i]
# 
#                 r = lRGBAColorAtPixel[0]
#                 g = lRGBAColorAtPixel[1]
#                 b = lRGBAColorAtPixel[2]
#                 # a = lRGBAColorAtPixel[3]
# 
#                 if (r != 0) or (g != 0) or (b != 0):   # or (a != 255):
#                     print "----- ===== ----- ===== ----- ===== ----- ===== ----- ===== r, g, b, a [i,j,k] = " + \
#                         str(r)+" "+str(g)+" "+str(b)+" "+str(a)+ \
#                         " ["+str(i)+" "+str(j)+" "+str(self.theCurrentIndex)+"] "
#                 self.imageSequenceArray[self.theCurrentIndex, j, i] = lRGBAColorAtPixel
#                 print i, j, self.theCurrentIndex, r,g,b
#                 if (lRGBAColorAtPixel.all() != (0,0,0).all()):
#                     print i, j, self.theCurrentIndex, lRGBAColorAtPixel
#         for i in xrange(0, lWidth, 1):
#             for j in xrange(0, lHeight, 1):
#                 lRGBAColorAtPixel = self.theCurrentImage.pixel(i,j) 
#                 lQColorAtPixel = QtGui.QColor( lRGBAColorAtPixel )
#                 r = lQColorAtPixel.red()
#                 g = lQColorAtPixel.green()
#                 b = lQColorAtPixel.blue()
#                 a = lQColorAtPixel.alpha()
#                 if (r != 0) or (g != 0) or (b != 0) or (a != 255):
#                     print "----- ===== ----- ===== ----- ===== ----- ===== ----- ===== r, g, b, a [i,j,k] = " + \
#                         str(r)+" "+str(g)+" "+str(b)+" "+str(a)+ \
#                         " ["+str(i)+" "+str(j)+" "+str(self.theCurrentIndex)+"] "
# 
#                 self.imageSequenceArray[self.theCurrentIndex, j, i] = lRGBAColorAtPixel
#         CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: theTrueSetCurrentImage() self.theCurrentIndex == "+str(self.theCurrentIndex) , CDConstants.DebugVerbose )





    # ------------------------------------------------------------------
    # 2011 - Mitja: setCurrentImageAndArrayLayer() loads a new
    #     QImage into self.theCurrentImage, and sets pixel values
    #     in one layer in the numpy array from self.theCurrentImage:
    # ------------------------------------------------------------------   
    def setCurrentImageAndArrayLayer(self, pImage):

        # skip parameters that are not proper QImage instances:
        if isinstance( pImage,  QtGui.QImage ) == False:
            return

        lTmpPixmap = QtGui.QPixmap.fromImage(pImage)
        self.theCurrentImage = QtGui.QImage(lTmpPixmap.toImage())

        lTmpArrayBGRA = self.qimage2numpy(self.theCurrentImage, "array")
        self.imageSequenceArray[self.theCurrentIndex] = lTmpArrayBGRA


#    for timing testing purposes, here's a possible code to time the above (now unused) function:
#
#         from timeit import Timer
#     
#         # print        Timer( self.theTrueSetCurrentImage )
# 
#         timerMeasureForFunction = Timer(self.theTrueSetCurrentImage)       # outside the try/except
#         try:
#             print timerMeasureForFunction.timeit(1)    # or timerMeasureForFunction.repeat(...)
#         except:
#             timerMeasureForFunction.print_exc()

#         CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: setCurrentImageAndArrayLayer() self.theCurrentIndex == "+str(self.theCurrentIndex) , CDConstants.DebugVerbose )

    # ------------------------------------------------------------------





    # ------------------------------------------------------------------
    # 2011 - Mitja: setCurrentFileName() justs sets the filename string into the
    #     "self.theCurrentIndex" key entry in the  self.imageSequenceFileNames  dict.
    # ------------------------------------------------------------------   
    def setCurrentFileName(self, pName):
        self.imageSequenceFileNames[self.theCurrentIndex] = pName




    # ------------------------------------------------------------------
    # 2011 - Mitja: setCurrentFileName() justs returns the filename string from the
    #     "self.theCurrentIndex" key entry in the  self.imageSequenceFileNames  dict.
    # ------------------------------------------------------------------   
    def getCurrentFileName(self):
        return str(self.imageSequenceFileNames[self.theCurrentIndex])




    # ------------------------------------------------------------------
    # 2011 - Mitja: setSequencePathName() sets the new path to 
    #    the directory containing all image files in the sequence:
    # ------------------------------------------------------------------   
    def setSequencePathName(self, pPathString):
        self.imageSequencePathString = pPathString




    # ------------------------------------------------------------------
    # Uses hashes of tuples to simulate 2-d arrays for the masks. 
    # ------------------------------------------------------------------
    def get_prewitt_masks(self): 
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
    def theTrueComputeCurrentEdge(self): 

        if ( self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUseDiscretizedToBWMode) == True ) :
            # if discretizing to black/white, the "volume array" values are to be used:

            # show a panel containing a progress bar:        
            lTmpProgressBarPanel=CDWaitProgressBar("Computing edge detection on volume.", self.theCurrentVolumeSliceImage.height())
            lTmpProgressBarPanel.show()
            lTmpProgressBarPanel.setRange(0, self.theCurrentVolumeSliceImage.height())
    
            # set the flag for the current edge array in the sequence as False, as we're computing it now:
            self.edgeInSequenceIsReadyFlags[self.theCurrentIndex] = False
    
            xmask, ymask = self.get_prewitt_masks() 
    
            width = self.theCurrentVolumeSliceImage.width()
            height = self.theCurrentVolumeSliceImage.height()
    
            # create a new greyscale image for the output 
    #         outimg = Image.new('L', (width, height)) 
    #         outpixels = list(outimg.getdata()) 
         
            for y in xrange(height): 
    
                lTmpProgressBarPanel.setValue(y)
    
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
                                r = int(self.volumeSequenceArray[self.theCurrentIndex, lY, lX, 0])
                                g = int(self.volumeSequenceArray[self.theCurrentIndex, lY, lX, 1])
                                b = int(self.volumeSequenceArray[self.theCurrentIndex, lY, lX, 2])
                                gray = (r + g + b) / 3
                                sumX += (gray) * xmask[i+1, j+1] 
         
                        for i in xrange(-1, 2): 
                            for j in xrange(-1, 2): 
                                lX = x+i
                                lY = y+j
                                # convolve the image pixels with the Prewitt mask,
                                #     approximating dI/dy 
                                r = int(self.volumeSequenceArray[self.theCurrentIndex, lY, lX, 0])
                                g = int(self.volumeSequenceArray[self.theCurrentIndex, lY, lX, 1])
                                b = int(self.volumeSequenceArray[self.theCurrentIndex, lY, lX, 2])
                                gray = (r + g + b) / 3
                                sumY += (gray) * ymask[i+1, j+1] 
         
                    # approximate the magnitude of the gradient 
                    magnitude = abs(sumX) + abs(sumY)
         
                    if magnitude > 255: magnitude = 255 
                    if magnitude < 0: magnitude = 0 
    
                    self.edgeSequenceArray[self.theCurrentIndex, y, x, 0] = numpy.uint8 (255 - magnitude)
                    self.edgeSequenceArray[self.theCurrentIndex, y, x, 1] = numpy.uint8 (255 - magnitude)
                    self.edgeSequenceArray[self.theCurrentIndex, y, x, 2] = numpy.uint8 (255 - magnitude)
    
            # set the flag for the current edge array in the sequence as True, as we're computing it now:
            self.edgeInSequenceIsReadyFlags[self.theCurrentIndex] = True
    
            lTmpProgressBarPanel.maxProgressBar()
            lTmpProgressBarPanel.accept()
            lTmpProgressBarPanel.close()
            
        else:
            # if NOT discretizing to black/white, the "image sequence array" values are to be used:

            # show a panel containing a progress bar:        
            lTmpProgressBarPanel=CDWaitProgressBar("Computing edge detection on images.", self.theCurrentImage.height())
            lTmpProgressBarPanel.show()
            lTmpProgressBarPanel.setRange(0, self.theCurrentImage.height())
    
            # set the flag for the current edge array in the sequence as False, as we're computing it now:
            self.edgeInSequenceIsReadyFlags[self.theCurrentIndex] = False
    
            xmask, ymask = self.get_prewitt_masks() 
    
            width = self.theCurrentImage.width()
            height = self.theCurrentImage.height()
    
            # create a new greyscale image for the output 
    #         outimg = Image.new('L', (width, height)) 
    #         outpixels = list(outimg.getdata()) 
         
            for y in xrange(height): 
    
                lTmpProgressBarPanel.setValue(y)
    
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
                                r = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 0])
                                g = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 1])
                                b = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 2])
                                gray = (r + g + b) / 3
                                sumX += (gray) * xmask[i+1, j+1] 
         
                        for i in xrange(-1, 2): 
                            for j in xrange(-1, 2): 
                                lX = x+i
                                lY = y+j
                                # convolve the image pixels with the Prewitt mask,
                                #     approximating dI/dy 
                                r = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 0])
                                g = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 1])
                                b = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 2])
                                gray = (r + g + b) / 3
                                sumY += (gray) * ymask[i+1, j+1] 
         
                    # approximate the magnitude of the gradient 
                    magnitude = abs(sumX) + abs(sumY)
         
                    if magnitude > 255: magnitude = 255 
                    if magnitude < 0: magnitude = 0 
    
                    self.edgeSequenceArray[self.theCurrentIndex, y, x, 0] = numpy.uint8 (255 - magnitude)
                    self.edgeSequenceArray[self.theCurrentIndex, y, x, 1] = numpy.uint8 (255 - magnitude)
                    self.edgeSequenceArray[self.theCurrentIndex, y, x, 2] = numpy.uint8 (255 - magnitude)
    
            # set the flag for the current edge array in the sequence as True, as we're computing it now:
            self.edgeInSequenceIsReadyFlags[self.theCurrentIndex] = True
    
            lTmpProgressBarPanel.maxProgressBar()
            lTmpProgressBarPanel.accept()
            lTmpProgressBarPanel.close()
        
        # end         if ( self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUseDiscretizedToBWMode) == True )



    # end of   def theTrueComputeCurrentEdge(self)
    # ------------------------------------------------------------------






    # ------------------------------------------------------------------
    # if we use numpy arrays for the masks,
    #   it's slower than tuples for indexes in the "manual" convolution code!!!
    #   So we can use numpy arrays for masks only for the optimized case.
    # ------------------------------------------------------------------
    def get_prewitt_array_masks(self): 
        xmask = numpy.zeros( (3, 3), dtype=numpy.int8 )
        ymask = numpy.zeros( (3, 3), dtype=numpy.int8 )
     
        xmask[0,0] = -1 
        xmask[0,1] = 0 
        xmask[0,2] = 1 
        xmask[1,0] = -1 
        xmask[1,1] = 0 
        xmask[1,2] = 1 
        xmask[2,0] = -1 
        xmask[2,1] = 0 
        xmask[2,2] = 1 
     
        ymask[0,0] = 1 
        ymask[0,1] = 1 
        ymask[0,2] = 1 
        ymask[1,0] = 0 
        ymask[1,1] = 0 
        ymask[1,2] = 0 
        ymask[2,0] = -1 
        ymask[2,1] = -1 
        ymask[2,2] = -1 
        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: get_prewitt_array_masks( xmask=" \
            + str(xmask) + ", ymask=" + str(ymask) + " )", CDConstants.DebugExcessive )
        return (xmask, ymask) 

    # ------------------------------------------------------------------
    # from http://stackoverflow.com/questions/2196693/improving-numpy-performance
    # ------------------------------------------------------------------
    def specialconvolve(a):
        # sorry, you must pad the input yourself
        rowconvol = a[1:-1,:] + a[:-2,:] + a[2:,:]
        colconvol = rowconvol[:,1:-1] + rowconvol[:,:-2] + rowconvol[:,2:] - 9*a[1:-1,1:-1]
        return colconvol

    # ------------------------------------------------------------------
    # http://en.wikipedia.org/wiki/Prewitt_operator
    # ------------------------------------------------------------------     
    def theSeparatedKernelComputeCurrentEdge(self): 


        # show a panel containing a progress bar:        
        lTmpProgressBarPanel=CDWaitProgressBar("Separated kernels computing edge detection.", self.theCurrentImage.height())
        lTmpProgressBarPanel.show()
        lTmpProgressBarPanel.setRange(0, self.theCurrentImage.height())

        # set the flag for the current edge array in the sequence as False, as we're computing it now:
        self.edgeInSequenceIsReadyFlags[self.theCurrentIndex] = False

        xmask, ymask = self.get_prewitt_array_masks() 

        width = self.theCurrentImage.width()
        height = self.theCurrentImage.height()

        # create a new greyscale image for the output 
#         outimg = Image.new('L', (width, height)) 
#         outpixels = list(outimg.getdata()) 

        # 1. do the sumX loop     
        for y in xrange(height): 

            lTmpProgressBarPanel.setValue(y)

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
                            r = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 0])
                            g = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 1])
                            b = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 2])
                            gray = (r + g + b) / 3
                            sumX += (gray) * xmask[i+1, j+1] 

# continue fix here in doing the 3 loops in a row!!!
     
                # the magnitude of the gradient would be using the root of the squares, we approximate:
                magnitude = abs(sumX) + abs(sumY)
     
                if magnitude > 255: magnitude = 255 
                if magnitude < 0: magnitude = 0 

                self.edgeSequenceArray[self.theCurrentIndex, y, x, 0] = numpy.uint8 (255 - magnitude)
                self.edgeSequenceArray[self.theCurrentIndex, y, x, 1] = numpy.uint8 (255 - magnitude)
                self.edgeSequenceArray[self.theCurrentIndex, y, x, 2] = numpy.uint8 (255 - magnitude)

        # 2. do the sumY loop     
        for y in xrange(height): 

            lTmpProgressBarPanel.setValue(y)

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
                            r = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 0])
                            g = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 1])
                            b = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 2])
                            gray = (r + g + b) / 3
                            sumX += (gray) * xmask[i+1, j+1] 
     
                    for i in xrange(-1, 2): 
                        for j in xrange(-1, 2): 
                            lX = x+i
                            lY = y+j
                            # convolve the image pixels with the Prewitt mask,
                            #     approximating dI/dy 
                            r = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 0])
                            g = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 1])
                            b = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 2])
                            gray = (r + g + b) / 3
                            sumY += (gray) * ymask[i+1, j+1] 
     
                # the magnitude of the gradient would be using the root of the squares, we approximate:
                magnitude = abs(sumX) + abs(sumY)
     
                if magnitude > 255: magnitude = 255 
                if magnitude < 0: magnitude = 0 

                self.edgeSequenceArray[self.theCurrentIndex, y, x, 0] = numpy.uint8 (255 - magnitude)
                self.edgeSequenceArray[self.theCurrentIndex, y, x, 1] = numpy.uint8 (255 - magnitude)
                self.edgeSequenceArray[self.theCurrentIndex, y, x, 2] = numpy.uint8 (255 - magnitude)


        # 3. do the magnitude loop
        for y in xrange(height): 

            lTmpProgressBarPanel.setValue(y)

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
                            r = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 0])
                            g = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 1])
                            b = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 2])
                            gray = (r + g + b) / 3
                            sumX += (gray) * xmask[i+1, j+1] 
     
                    for i in xrange(-1, 2): 
                        for j in xrange(-1, 2): 
                            lX = x+i
                            lY = y+j
                            # convolve the image pixels with the Prewitt mask,
                            #     approximating dI/dy 
                            r = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 0])
                            g = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 1])
                            b = int(self.imageSequenceArray[self.theCurrentIndex, lY, lX, 2])
                            gray = (r + g + b) / 3
                            sumY += (gray) * ymask[i+1, j+1] 
     
                # the magnitude of the gradient would be using the root of the squares, we approximate:
                magnitude = abs(sumX) + abs(sumY)
     
                if magnitude > 255: magnitude = 255 
                if magnitude < 0: magnitude = 0 

                self.edgeSequenceArray[self.theCurrentIndex, y, x, 0] = numpy.uint8 (255 - magnitude)
                self.edgeSequenceArray[self.theCurrentIndex, y, x, 1] = numpy.uint8 (255 - magnitude)
                self.edgeSequenceArray[self.theCurrentIndex, y, x, 2] = numpy.uint8 (255 - magnitude)


        # set the flag for the current edge array in the sequence as True, as we're computing it now:
        self.edgeInSequenceIsReadyFlags[self.theCurrentIndex] = True

        lTmpProgressBarPanel.maxProgressBar()
        lTmpProgressBarPanel.accept()
        lTmpProgressBarPanel.close()
        

    # end of   def theSeparatedKernelComputeCurrentEdge(self)
    # ------------------------------------------------------------------





    # ------------------------------------------------------------------
    # 2011 - Mitja: computeCurrentEdge() computes edge detection on the current image:
    # ------------------------------------------------------------------   
    def computeCurrentEdge(self):

        # only really compute the current edge if it hasn't been computed yet:
        if (self.edgeInSequenceIsReadyFlags[self.theCurrentIndex] == False):
            
            # adjusting Timer() setup for Python 2.5:
            if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
                timerMeasureForFunction = Timer(self.theTrueComputeCurrentEdge)  # define it before the try/except
                try:
                    lTheTimeItTook = timerMeasureForFunction.timeit(1)           # or timerMeasureForFunction.repeat(...)
                    CDConstants.printOut( str(lTheTimeItTook)+ \
                        " seconds it took theTrueComputeCurrentEdge(), self.theCurrentIndex == " + \
                        str(self.theCurrentIndex) + " in CDImageSequence: computeCurrentEdge()" , CDConstants.DebugVerbose )
                except:
                    CDConstants.printOut( "CDImageSequence: computeCurrentEdge() code exception!   self.theCurrentIndex == "+str(self.theCurrentIndex) , CDConstants.DebugSparse )
                    timerMeasureForFunction.print_exc()
                    CDConstants.printOut( "CDImageSequence: computeCurrentEdge() code exception!   self.theCurrentIndex == "+str(self.theCurrentIndex) , CDConstants.DebugSparse )
            else:
                # timerMeasureForFunction = Timer('theTrueComputeCurrentEdge()', 'from CDImageSequence import theTrueComputeCurrentEdge')  # define it before the try/except
                self.theTrueComputeCurrentEdge()


    # end of def computeCurrentEdge(self)
    # ------------------------------------------------------------------










    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def theTrueComputeContours(self): 

        if ( self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUseDiscretizedToBWMode) == True ) :
            # ------------------------------------------------------------------
            # if discretizing to black/white, the "volume array" values are to be used:
            # ------------------------------------------------------------------

            # set the flag for the current edge array in the sequence as False, as we're computing it now:
            self.contoursAreReadyFlag = False
    
            lXmask, lYmask = self.get_prewitt_masks() 
    
            lWidth = self.sizeX
            lHeight = self.sizeY
            lDepth = self.sizeZ
    
            # show a panel containing a progress bar:        
            lTmpProgressBarPanel=CDWaitProgressBarWithImage("Computing 3D Contours edge detection from Detected B/W Volume.", self.theCurrentImage.height())
            lTmpProgressBarPanel.show()
            lTmpProgressBarPanel.setRange(0, self.theCurrentImage.height())
    
            lPixmap = QtGui.QPixmap( lDepth, lHeight)
            lPixmap.fill(QtCore.Qt.transparent)

            # store the pixmap holding the specially rendered scene:
            lTmpProgressBarPanel.theProgressBarImageLabel.setPixmap(lPixmap)
            lTmpProgressBarPanel.theProgressBarImageLabel.image = lPixmap.toImage()    
            lTmpProgressBarPanel.theProgressBarImageLabel.width = int( lPixmap.width() )
            lTmpProgressBarPanel.theProgressBarImageLabel.height = int ( lPixmap.height() )

            if ( lTmpProgressBarPanel.theContentWidget.width() < (lDepth + 20) ):
                lTheNewWidth = lDepth + 20
            else:
                lTheNewWidth = lTmpProgressBarPanel.theContentWidget.width()
            if ( lTmpProgressBarPanel.theContentWidget.height() < (lHeight + 20) ):
                lTheNewHeight = lHeight + 20
            else:
                lTheNewHeight = lTmpProgressBarPanel.theContentWidget.height()
            lTmpProgressBarPanel.theContentWidget.resize(lTheNewWidth, lTheNewHeight)
            lTmpProgressBarPanel.theContentWidget.update()
            lTmpProgressBarPanel.adjustSize()
    
            # -------------------------------
            # scan across x-direction layers:
            # -------------------------------
            for x in xrange(lWidth): 

                # prepare a QPainter for visual feedback of computed contours:
                lTmpPainter = QtGui.QPainter(lPixmap)
                lTmpPen = QtGui.QPen(QtCore.Qt.black)
                lTmpPen.setWidth(2)
                lTmpPen.setCosmetic(True)
                lTmpPainter.setPen(lTmpPen)
                lTmpBrush = QtGui.QBrush(QtGui.QColor(QtCore.Qt.red))
                lTmpPainter.setBrush(lTmpBrush)

        
                lTmpProgressBarPanel.setTitle( self.tr(" Scanning x layer %1 of %2 from computed B/W Volume \n to generate 3D Contour-boundary points ... ").arg( \
                    str(x) ).arg( str(lWidth) )  ) 
    
                CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: self.theTrueComputeContours() lPixmap w,h =" + \
                      str(lTmpProgressBarPanel.theProgressBarImageLabel.width) + " " + str(lTmpProgressBarPanel.theProgressBarImageLabel.height) + \
                      " Scanning x layer "+str(x)+" of "+str(lWidth)+" from computed B/W Volume to generate 3D Contour-boundary points.", CDConstants.DebugVerbose )
        
                # adjusts the size of the label widget to fit its contents (i.e. the pixmap):
                lTmpProgressBarPanel.theProgressBarImageLabel.adjustSize()
                lTmpProgressBarPanel.theProgressBarImageLabel.show()
                lTmpProgressBarPanel.theProgressBarImageLabel.update()
             
                # -----------------------------
                # scan across y-direction rows:
                # -----------------------------
                for y in xrange(lHeight): 
        
                    # provide visual feedback to user:
                    lTmpProgressBarPanel.setValue(y)
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
                                    r = int(self.volumeSequenceArray[lZ, lY, x, 0])
                                    g = int(self.volumeSequenceArray[lZ, lY, x, 1])
                                    b = int(self.volumeSequenceArray[lZ, lY, x, 2])
                                    gray = (r + g + b) / 3
                                    sumZ += (gray) * lXmask[k+1, j+1] 
             
                            for k in xrange(-1, 2): 
                                for j in xrange(-1, 2): 
                                    lZ = z+k
                                    lY = y+j
                                    # convolve the image pixels with the Prewitt mask,
                                    #     approximating dI/dy 
                                    r = int(self.volumeSequenceArray[lZ, lY, x, 0])
                                    g = int(self.volumeSequenceArray[lZ, lY, x, 1])
                                    b = int(self.volumeSequenceArray[lZ, lY, x, 2])
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
                lTmpProgressBarPanel.theProgressBarImageLabel.drawPixmapAtPoint(lPixmap)
                lTmpProgressBarPanel.theProgressBarImageLabel.update()

    
            # ------------------------------------------
            # <-- end of scan across x-direction layers.
            # ------------------------------------------
        
            # set the flag for the contours array in the sequence as True, as we've just computed it:
            self.contoursAreReadyFlag = True
        
            lTmpProgressBarPanel.maxProgressBar()
            lTmpProgressBarPanel.accept()
            lTmpProgressBarPanel.close()
        # ------------------------------------------------------------------
        # end of  if ( self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUseDiscretizedToBWMode) == True )
        # i.e. if discretizing to black/white, the "volume array" values have been used <====
        # ------------------------------------------------------------------
        else:
            # ------------------------------------------------------------------
            # if NOT discretizing to black/white, the "image sequence array" values are to be used:
            # ------------------------------------------------------------------

            # set the flag for the current edge array in the sequence as False, as we're computing it now:
            self.contoursAreReadyFlag = False
    
            lXmask, lYmask = self.get_prewitt_masks() 
    
            lWidth = self.sizeX
            lHeight = self.sizeY
            lDepth = self.sizeZ
    
            # show a panel containing a progress bar:        
            lTmpProgressBarPanel=CDWaitProgressBarWithImage("Computing 3D Contours edge detection from Image Sequence.", self.theCurrentImage.height())
            lTmpProgressBarPanel.show()
            lTmpProgressBarPanel.setRange(0, self.theCurrentImage.height())
    
            lPixmap = QtGui.QPixmap( lDepth, lHeight)
            lPixmap.fill(QtCore.Qt.transparent)

            # store the pixmap holding the specially rendered scene:
            lTmpProgressBarPanel.theProgressBarImageLabel.setPixmap(lPixmap)
            lTmpProgressBarPanel.theProgressBarImageLabel.image = lPixmap.toImage()    
            lTmpProgressBarPanel.theProgressBarImageLabel.width = int( lPixmap.width() )
            lTmpProgressBarPanel.theProgressBarImageLabel.height = int ( lPixmap.height() )

            if ( lTmpProgressBarPanel.theContentWidget.width() < (lDepth + 20) ):
                lTheNewWidth = lDepth + 20
            else:
                lTheNewWidth = lTmpProgressBarPanel.theContentWidget.width()
            if ( lTmpProgressBarPanel.theContentWidget.height() < (lHeight + 20) ):
                lTheNewHeight = lHeight + 20
            else:
                lTheNewHeight = lTmpProgressBarPanel.theContentWidget.height()
            lTmpProgressBarPanel.theContentWidget.resize(lTheNewWidth, lTheNewHeight)
            lTmpProgressBarPanel.theContentWidget.update()
            lTmpProgressBarPanel.adjustSize()
    
            # -------------------------------
            # scan across x-direction layers:
            # -------------------------------
            for x in xrange(lWidth): 

                # prepare a QPainter for visual feedback of computed contours:
                lTmpPainter = QtGui.QPainter(lPixmap)
                lTmpPen = QtGui.QPen(QtCore.Qt.black)
                lTmpPen.setWidth(2)
                lTmpPen.setCosmetic(True)
                lTmpPainter.setPen(lTmpPen)
                lTmpBrush = QtGui.QBrush(QtGui.QColor(QtCore.Qt.red))
                lTmpPainter.setBrush(lTmpBrush)

        
                lTmpProgressBarPanel.setTitle( self.tr(" Scanning x layer %1 of %2 from Image Sequence Volume \n to generate 3D Contour-boundary points... ").arg( \
                    str(x) ).arg( str(lWidth) )  ) 
    
                CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: self.theTrueComputeContours() lPixmap w,h =" + \
                      str(lTmpProgressBarPanel.theProgressBarImageLabel.width) + " " + str(lTmpProgressBarPanel.theProgressBarImageLabel.height) + \
                      " Scanning x layer "+str(x)+" of "+str(lWidth)+" from Image Sequence Volume to generate 3D Contour-boundary points.", CDConstants.DebugVerbose )
        
                # adjusts the size of the label widget to fit its contents (i.e. the pixmap):
                lTmpProgressBarPanel.theProgressBarImageLabel.adjustSize()
                lTmpProgressBarPanel.theProgressBarImageLabel.show()
                lTmpProgressBarPanel.theProgressBarImageLabel.update()
             
                # -----------------------------
                # scan across y-direction rows:
                # -----------------------------
                for y in xrange(lHeight): 
        
                    # provide visual feedback to user:
                    lTmpProgressBarPanel.setValue(y)
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
                lTmpProgressBarPanel.theProgressBarImageLabel.drawPixmapAtPoint(lPixmap)
                lTmpProgressBarPanel.theProgressBarImageLabel.update()

    
            # ------------------------------------------
            # <-- end of scan across x-direction layers.
            # ------------------------------------------
        
            # set the flag for the contours array in the sequence as True, as we've just computed it:
            self.contoursAreReadyFlag = True
        
            lTmpProgressBarPanel.maxProgressBar()
            lTmpProgressBarPanel.accept()
            lTmpProgressBarPanel.close()

        # ------------------------------------------------------------------
        # end of else to  if ( self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUseDiscretizedToBWMode) == True )
        # i.e. if NOT discretizing to black/white, the "image sequence array" values have been used <===
        # ------------------------------------------------------------------

    # end of   def theTrueComputeContours(self)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------





    # ------------------------------------------------------------------
    # 2011 - Mitja: computeContours() computes edge detection on all images:
    # ------------------------------------------------------------------   
    def computeContours(self):

        # only really compute the contours if they haven't been computed yet:
        if (self.contoursAreReadyFlag == False):
            
            # adjusting Timer() setup for Python 2.5:
            if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
                timerMeasureForFunction = Timer(self.theTrueComputeContours)  # define it before the try/except
                try:
                    lTheTimeItTook = timerMeasureForFunction.timeit(1)        # or timerMeasureForFunction.repeat(...)
                    CDConstants.printOut( str(lTheTimeItTook)+ \
                        " seconds it took theTrueComputeContours() in CDImageSequence: computeContours()" , CDConstants.DebugVerbose )
                except:
                    CDConstants.printOut( "CDImageSequence: computeContours() code exception! " , CDConstants.DebugSparse )
                    timerMeasureForFunction.print_exc()
                    CDConstants.printOut( "CDImageSequence: computeContours() code exception! " , CDConstants.DebugSparse )
            else:
                # timerMeasureForFunction = Timer('theTrueComputeContours()', 'from CDImageSequence import theTrueComputeContours')  # define it before the try/except
                self.theTrueComputeContours()

        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: computeContours() self.theCurrentIndex == "+str(self.theCurrentIndex) , CDConstants.DebugVerbose )


    # end of  def computeContours(self)
    # ------------------------------------------------------------------   







    # ------------------------------------------------------------------
    # 2011 - Mitja: assignAllProcessingModesForImageSequenceToPIFF()
    #    sets all binary-flags in a class global keeping track of the modes
    #    for generating PIFF from displayed imported image sequence
    # ------------------------------------------------------------------   
    def assignAllProcessingModesForImageSequenceToPIFF(self, pValue):

        # bin() does not exist in Python 2.5:
        if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: assignAllProcessingModesForImageSequenceToPIFF() bin(self.theProcessingModeForImageSequenceToPIFF) == " + \
                str(bin(self.theProcessingModeForImageSequenceToPIFF)) + " bin(pValue) == " + str(bin(pValue)), CDConstants.DebugVerbose )
        else:
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: assignAllProcessingModesForImageSequenceToPIFF() self.theProcessingModeForImageSequenceToPIFF == " + \
                str(self.theProcessingModeForImageSequenceToPIFF) + " pValue == " + str(pValue), CDConstants.DebugVerbose )

        # if we are changing i.e. toggling B/W discretization mode,  invalidate all edges and contours computed so far:
        if  ( ( pValue & (1 << CDConstants.ImageSequenceUseDiscretizedToBWMode) )  and \
                ( self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUseDiscretizedToBWMode) == False ) )  \
            or \
            ( (not (pValue & (1 << CDConstants.ImageSequenceUseDiscretizedToBWMode)))  and \
                (self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUseDiscretizedToBWMode) == True )  ):

            self.edgeInSequenceIsReadyFlags = numpy.zeros( (self.sizeZ), dtype=numpy.bool )
            self.contoursAreReadyFlag = False
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: assignAllProcessingModesForImageSequenceToPIFF() self.contoursAreReadyFlag=="+str(self.contoursAreReadyFlag)+ \
                "self.edgeInSequenceIsReadyFlags"+str(self.edgeInSequenceIsReadyFlags) , CDConstants.DebugVerbose )

        # assign the actual mode:
        self.theProcessingModeForImageSequenceToPIFF = pValue

        # then check if the image sequence has to display its edge or its contours,
        #    and if so: ask the sequence to compute the edge or contours if not ready yet:

        if (self.theProcessingModeForImageSequenceToPIFF & (1 << CDConstants.ImageSequenceUseEdge) ):
            self.computeCurrentEdge()

        if (self.theProcessingModeForImageSequenceToPIFF & (1 << CDConstants.ImageSequenceUse3DContours) ):
            self.computeContours()

        # bin() does not exist in Python 2.5:
        if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: assignAllProcessingModesForImageSequenceToPIFF() bin(self.theProcessingModeForImageSequenceToPIFF) == "+str(bin(self.theProcessingModeForImageSequenceToPIFF)) , CDConstants.DebugVerbose )
        else:
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: assignAllProcessingModesForImageSequenceToPIFF() self.theProcessingModeForImageSequenceToPIFF == "+str(self.theProcessingModeForImageSequenceToPIFF) , CDConstants.DebugVerbose )

    # end of   def assignAllProcessingModesForImageSequenceToPIFF(self, pValue)
    # ------------------------------------------------------------------   






    # ------------------------------------------------------------------
    # 2011 - Mitja: enableAProcessingModeForImageSequenceToPIFF()
    #    sets (to 1 AKA True) a binary-flag in a class global keeping track of the modes
    #    for generating PIFF from displayed imported image sequence
    # ------------------------------------------------------------------   
    def enableAProcessingModeForImageSequenceToPIFF(self, pValue):

        # if we are changing choice on discretization to B/W mode,  invalidate all computed edges and contours:
        if ( (pValue == CDConstants.ImageSequenceUseDiscretizedToBWMode) and \
            (self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUseDiscretizedToBWMode) == False) ):

            self.edgeInSequenceIsReadyFlags = numpy.zeros( (self.sizeZ), dtype=numpy.bool )
            self.contoursAreReadyFlag = False


        # do a bitwise-OR with pValue, to set the specific bit to 1 and leave other bits unchanged:
        self.theProcessingModeForImageSequenceToPIFF |= (1 << pValue)

        # bin() does not exist in Python 2.5:
        if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: enableAProcessingModeForImageSequenceToPIFF() bin(self.theProcessingModeForImageSequenceToPIFF) == "+str(bin(self.theProcessingModeForImageSequenceToPIFF))+" from pValue =="+str(pValue) , CDConstants.DebugVerbose )
        else:
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: enableAProcessingModeForImageSequenceToPIFF() self.theProcessingModeForImageSequenceToPIFF == "+str(self.theProcessingModeForImageSequenceToPIFF)+" from pValue =="+str(pValue) , CDConstants.DebugVerbose )

        CDConstants.printOut("[A] hello, I'm "+str(debugWhoIsTheRunningFunction())+", parent is "+str(debugWhoIsTheParentFunction())+ \
            " ||||| self.repaintEventsCounter=="+str(self.repaintEventsCounter), CDConstants.DebugTODO )

    # end of  def enableAProcessingModeForImageSequenceToPIFF(self, pValue)
    # ------------------------------------------------------------------   




    # ------------------------------------------------------------------
    # 2011 - Mitja: resetToOneProcessingModeForImageSequenceToPIFF()
    #    sets (to 1 AKA True) a binary-flag in a class global keeping track of the modes
    #    for generating PIFF from displayed imported image sequence
    # ------------------------------------------------------------------
    def resetToOneProcessingModeForImageSequenceToPIFF(self, pValue):

        # if we are changing choice on discretization to B/W mode,  invalidate all computed edges and contours:
        if (    (pValue == CDConstants.ImageSequenceUseDiscretizedToBWMode) and \
                (self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUseDiscretizedToBWMode) == False)  ) \
            or \
            (   (pValue != CDConstants.ImageSequenceUseDiscretizedToBWMode) and \
                (self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUseDiscretizedToBWMode) == True)  ):
            # if we are changing choice on discretization to B/W mode,  invalidate all computed edges and contours:
            self.edgeInSequenceIsReadyFlags = numpy.zeros( (self.sizeZ), dtype=numpy.bool )
            self.contoursAreReadyFlag = False

        # set bitwise pValue, to set only the specific bit to 1 and all other bits to 0:
        self.theProcessingModeForImageSequenceToPIFF = (1 << pValue)

        # bin() does not exist in Python 2.5:
        if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: resetToOneProcessingModeForImageSequenceToPIFF() bin(self.theProcessingModeForImageSequenceToPIFF) == "+str(bin(self.theProcessingModeForImageSequenceToPIFF))+" from pValue =="+str(pValue) , CDConstants.DebugVerbose )
        else:
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: resetToOneProcessingModeForImageSequenceToPIFF() self.theProcessingModeForImageSequenceToPIFF == "+str(self.theProcessingModeForImageSequenceToPIFF)+" from pValue =="+str(pValue) , CDConstants.DebugVerbose )

    # end of  def resetToOneProcessingModeForImageSequenceToPIFF(self, pValue)
    # ------------------------------------------------------------------   




    # ------------------------------------------------------------------
    # 2011 - Mitja: disableAProcessingModeForImageSequenceToPIFF()
    #    clears (to 0 AKA True) a binary-flag in a class global keeping track of the modes
    #    for generating PIFF from displayed imported image sequence
    # ------------------------------------------------------------------   
    def disableAProcessingModeForImageSequenceToPIFF(self, pValue):

        # if we are changing choice on discretization to B/W mode,  invalidate all computed edges and contours:
        if ( (pValue == CDConstants.ImageSequenceUseDiscretizedToBWMode) and \
            (self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUseDiscretizedToBWMode) == True) ):
            # if we are changing choice on discretization to B/W mode,  invalidate all computed edges and contours:
            self.edgeInSequenceIsReadyFlags = numpy.zeros( (self.sizeZ), dtype=numpy.bool )
            self.contoursAreReadyFlag = False

        # do a bitwise-AND with negated pValue, to clear the specific bit to 0:
        self.theProcessingModeForImageSequenceToPIFF &= ~(1 << pValue)

        # bin() does not exist in Python 2.5:
        if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: disableAProcessingModeForImageSequenceToPIFF() bin(self.theProcessingModeForImageSequenceToPIFF) == "+str(bin(self.theProcessingModeForImageSequenceToPIFF))+" from pValue =="+str(pValue) , CDConstants.DebugVerbose )
        else:
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: disableAProcessingModeForImageSequenceToPIFF() self.theProcessingModeForImageSequenceToPIFF == "+str(self.theProcessingModeForImageSequenceToPIFF)+" from pValue =="+str(pValue) , CDConstants.DebugVerbose )

    #  def disableAProcessingModeForImageSequenceToPIFF(self, pValue)
    # ------------------------------------------------------------------   




    # ------------------------------------------------------------------
    # 2011 - Mitja: getAProcessingModeStatusForImageSequenceToPIFF()
    #    returns a Boolean from the binary-flag in a class global keeping track of the modes
    #    for generating PIFF from displayed imported image sequence
    # ------------------------------------------------------------------   
    def getAProcessingModeStatusForImageSequenceToPIFF(self, pValue):
        if ( self.theProcessingModeForImageSequenceToPIFF & (1 << pValue) ):
            # bin() does not exist in Python 2.5:
            if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
                CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: getAProcessingModeStatusForImageSequenceToPIFF() TRUE bin(pValue, (1 << pValue)) == "+str(pValue)+" , "+str(bin(1 << pValue)) , CDConstants.DebugVerbose )
            else:
                CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: getAProcessingModeStatusForImageSequenceToPIFF() TRUE (pValue, (1 << pValue) == "+str(pValue)+" , "+str(1 << pValue) , CDConstants.DebugVerbose )
            return True
        else:
            # bin() does not exist in Python 2.5:
            if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
                CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: getAProcessingModeStatusForImageSequenceToPIFF() FALSE bin(pValue, (1 << pValue)) == "+str(pValue)+" , "+str(bin(1 << pValue)) , CDConstants.DebugVerbose )
            else:
                CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: getAProcessingModeStatusForImageSequenceToPIFF() FALSE pValue, (1 << pValue) == "+str(pValue)+" , "+str(1 << pValue) , CDConstants.DebugVerbose )
            return False




    # ------------------------------------------------------------------
    # 2011 - Mitja: setCurrentIndexWithoutUpdatingGUI() is to set the current image index within the sequence,
    #   during importing images and other non-GUI tasks:
    # ------------------------------------------------------------------   
    def setCurrentIndexWithoutUpdatingGUI(self, pValue):
        self.theCurrentIndex = pValue

        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: setCurrentIndexWithoutUpdatingGUI() self.theCurrentIndex == "+str(self.theCurrentIndex) , CDConstants.DebugVerbose )
    # end of    def setCurrentIndexWithoutUpdatingGUI(self, pValue)
    # ------------------------------------------------------------------   



    # ------------------------------------------------------------------
    # 2011 - Mitja: setCurrentIndex() is to set the current image index within the sequence:
    # ------------------------------------------------------------------   
    def setCurrentIndex(self, pValue):
        self.theCurrentIndex = pValue

        CDConstants.printOut(  "___ - DEBUG ----- CDImageSequence: setCurrentIndex()  --  1.   self.theCurrentIndex=="+str(self.theCurrentIndex), CDConstants.DebugTODO )
        # TODO remove TODO time.sleep(1.0)

        #  create images... theCurrentImage and theCurrentEdge and theCurrentVolumeSliceImage from the current layer in the sequence arrays:
        # these images are now *not* painted here, but from setCurrentIndex() ...
        # ... should *not* call imageCurrentImage() from paintTheImageSequence(),
        #    because paintTheImageSequence() is part of the repainting and ought not open additional widgets or cause repaints...
        #    (and imageCurrentImage() may open dialog boxes etc.)
        self.imageCurrentImage()


        CDConstants.printOut(  "___ - DEBUG ----- CDImageSequence: setCurrentIndex()  --  2.   self.imageCurrentImage() DONE", CDConstants.DebugTODO )
        # TODO remove TODO time.sleep(1.0)
                

        # emit a signal to update image sequence size GUI controls:
        lDict = { \
            0: str(self.theCurrentIndex), \
            1: str(self.imageSequenceFileNames[self.theCurrentIndex]), \
            }

        self.signalThatCurrentIndexSet.emit(lDict)        

        CDConstants.printOut(  "___ - DEBUG ----- CDImageSequence: setCurrentIndex()  --  3.   self.signalThatCurrentIndexSet.emit( lDict=="+str(lDict)+" )", CDConstants.DebugTODO )
        # TODO remove TODO time.sleep(1.0)
        
#         CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: setCurrentIndex() self.theCurrentIndex == "+str(self.theCurrentIndex) , CDConstants.DebugVerbose )
    # end of    def setCurrentIndex(self, pValue)
    # ------------------------------------------------------------------   




    # ------------------------------------------------------------------
    # 2011 - Mitja: getCurrentIndex() is to get the current image index within the sequence:
    # ------------------------------------------------------------------   
    def getCurrentIndex(self):
#         CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: getCurrentIndex() self.theCurrentIndex = "+str(self.theCurrentIndex) , CDConstants.DebugExcessive )
        return (self.theCurrentIndex)





    # ------------------------------------------------------------------
    # 2011 - Mitja: setSequenceCurrentColor():
    # ------------------------------------------------------------------   
    def setSequenceCurrentColor(self, pValue):
        self.theImageSequenceColor = pValue


    # ------------------------------------------------------------------
    # 2011 - Mitja: getSequenceCurrentColor():
    # ------------------------------------------------------------------   
    def getSequenceCurrentColor(self):
        return (self.theImageSequenceColor)





    # ------------------------------------------------------------------
    # 2011 - Mitja: setSequenceWallColor():
    # ------------------------------------------------------------------   
    def setSequenceWallColor(self, pValue):
        self.theImageSequenceWallColor = pValue


    # ------------------------------------------------------------------
    # 2011 - Mitja: getSequenceWallColor():
    # ------------------------------------------------------------------   
    def getSequenceWallColor(self):
        return (self.theImageSequenceWallColor)




    # ------------------------------------------------------------------
    # 2011 - Mitja: the setSequenceLoadedFromFiles() function is to mimic
    #    the behavior of our CDImageLayer class:
    # ------------------------------------------------------------------   
    def setSequenceLoadedFromFiles(self, pTrueOrFalse):
        self.imageSequenceLoaded = pTrueOrFalse
        if isinstance( self.graphicsSceneWindow, QtGui.QWidget ) == True:
            self.graphicsSceneWindow.scene.update()
        CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: setSequenceLoadedFromFiles( " \
            + str(self.imageSequenceLoaded) + " )", CDConstants.DebugVerbose )


    # ------------------------------------------------------------------
    # 2011 - Mitja: the setMouseTracking() function is to mimic a QLabel,
    #   which being a QWidget supports setMouseTracking(). Instead, we
    #   pass it upstream to the parent widget's QGraphicsScene:
    # ------------------------------------------------------------------   
    def setMouseTracking(self, pTrueOrFalse):
        if isinstance( self.graphicsSceneWindow, QtGui.QWidget ) == True:
            self.graphicsSceneWindow.view.setMouseTracking(pTrueOrFalse)


    # ------------------------------------------------------------------
    # 2011 - Mitja: the setWidthOfFixedRaster() function is to set the fixed raster width:
    # ------------------------------------------------------------------   
    def setWidthOfFixedRaster(self, pGridWidth):
        self.fixedRasterWidth = pGridWidth



    # ------------------------------------------------------------
    # 2011 - Mitja: provide color-to-color distance calculation:
    # ------------------------------------------------------------
    def colorToColorIsCloseDistance(self, pC1, pC2, pDist):
        r1 = QColor(pC1).redF()
        g1 = QColor(pC1).greenF()
        b1 = QColor(pC1).blueF()
        r2 = QColor(pC2).redF()
        g2 = QColor(pC2).greenF()
        b2 = QColor(pC2).blueF()
       
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
        r1 = QColor(pC1).redF()
        g1 = QColor(pC1).greenF()
        b1 = QColor(pC1).blueF()
        r2 = QColor(pC2).redF()
        g2 = QColor(pC2).greenF()
        b2 = QColor(pC2).blueF()
       
        dr = (r2-r1) * (r2-r1)
        dg = (g2-g1) * (g2-g1)
        db = (b2-b1) * (b2-b1)
        if (dr < 0.0001) and (dg < 0.0001) and (db < 0.0001) :
#             print "TRUE",  r1, r2, g1, g2, b1, b2
            return True
        else:
#             print "FALSE",  r1, r2, g1, g2, b1, b2
            return False



    # ------------------------------------------------------------------
    # 2011 - Mitja: paintTheImageSequence() paints/draws all that needs to go
    #   into the Image Sequence, and may be called directly or by our paintEvent() handler:
    # ------------------------------------------------------------------   
    def paintTheImageSequence(self, pThePainter):
    
#         CDConstants.printOut("     =====     =====     =====     =====     ", CDConstants.DebugTODO )
#         lTmpLengthI = len (inspect.stack())
#         lTmpStrI = " >====> stack:"
#         CDConstants.printOut("in paintTheImageSequence(), len (inspect.stack()) = "+str(lTmpLengthI), CDConstants.DebugTODO )        
#         for i in xrange(lTmpLengthI):
#             lTmpLengthJ = len (inspect.stack()[i])
#             lTmpStrJ = ""
#             for j in xrange(lTmpLengthJ):
#                 lTmpStrJ = lTmpStrJ + "["+str(j)+"]=="+str(inspect.stack()[i][j])+" "
#             lTmpStrI = lTmpStrI + "\n["+str(i)+"] ==  "+lTmpStrJ
#         CDConstants.printOut(lTmpStrI, CDConstants.DebugTODO )
#         
#         CDConstants.printOut("     =====     =====     =====     =====     ", CDConstants.DebugTODO )
#         traceback.print_stack()
# 
#         CDConstants.printOut("     =====     =====     =====     =====     ", CDConstants.DebugTODO )
# 2012 - Mitja: advanced debugging tools,
#   work only on Posix-compliant systems so far (no MS-Windows)
#   commented out for now:
#         os.kill(os.getpid(), signal.SIGUSR1)
# 
#         CDConstants.printOut("     =====     =====     =====     =====     ", CDConstants.DebugTODO )

        CDConstants.printOut("[B] hello, I'm "+str(debugWhoIsTheRunningFunction())+", parent is "+str(debugWhoIsTheParentFunction())+ \
            " ||||| self.repaintEventsCounter=="+str(self.repaintEventsCounter)+ \
            " ||||| CDImageSequence.paintTheImageSequence(pThePainter=="+str(pThePainter)+")", CDConstants.DebugTODO )


        # CDConstants.printOut(  "paintTheImageSequence()  --  1.", CDConstants.DebugTODO )
        # TODO remove TODO time.sleep(1.0)


        # paint into the passed QPainter parameter (this comes from drawForeground() in DiagramScene) :
        lPainter = pThePainter
        # the QPainter has to be passed with begin() already called on it:
        # lPainter.begin()


        # CDConstants.printOut(  "paintTheImageSequence()  --  2.", CDConstants.DebugTODO )
        # TODO remove TODO time.sleep(1.0)







        # push the QPainter's current state onto a stack, to be followed by a restore() below:
        lPainter.save()




        # CDConstants.printOut(  "paintTheImageSequence()  --  3.", CDConstants.DebugTODO )
        # TODO remove TODO time.sleep(1.0)



# TODO: genera le immagini separatamente per input ("seeds"), edge, 3D contours e volume, con trasparenze e colori come da bottoni,
# e abilita in ordine ma sposta volume prima de edge e 3D contours, e sposta 3D contours prima de edge.
# 
# 
# Infine: verifica che salvi tutto e nell'ordine giusto come spiega' sopra.


        #  create images... theCurrentImage and theCurrentEdge and theCurrentVolumeSliceImage from the current layer in the sequence arrays:
        # these images are now *not* painted here, but from setCurrentIndex() ...
        # ... should *not* call imageCurrentImage() from paintTheImageSequence(),
        #    because paintTheImageSequence() is part of the repainting and ought not open additional widgets or cause repaints...
        #    (and imageCurrentImage() may open dialog boxes etc.)
        # self.imageCurrentImage()


        # CDConstants.printOut(  "paintTheImageSequence()  --  4.", CDConstants.DebugTODO )
        # TODO remove TODO time.sleep(1.0)


        # draw image's full area regions, their computed edges, etc. according to the users' choosen GUI buttons:
        #
        if ( self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUseAreaSeeds) ) :

            # draw the selected image, if there is one:
            if isinstance( self.theCurrentImage, QtGui.QImage ) == True:
                lPixMap = QtGui.QPixmap.fromImage(self.theCurrentImage)
                lPainter.drawPixmap(QtCore.QPoint(0,0), lPixMap)
                CDConstants.printOut(  "paintTheImageSequence()  --  4b. --  CDConstants.ImageSequenceUseAreaSeeds TRUE, painted:  self.theCurrentImage", CDConstants.DebugTODO )


        # CDConstants.printOut(  "paintTheImageSequence()  --  5.", CDConstants.DebugTODO )
        # TODO remove TODO time.sleep(1.0)


        if ( self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUse3DVolume) ) :

            # draw the selected volume slice image, if there is one:
            if isinstance( self.theCurrentVolumeSliceImage, QtGui.QImage ) == True:
                lPixMap = QtGui.QPixmap.fromImage(self.theCurrentVolumeSliceImage)
                lPainter.drawPixmap(QtCore.QPoint(0,0), lPixMap)
                CDConstants.printOut(  "paintTheImageSequence()  --  5b. --  CDConstants.ImageSequenceUse3DVolume TRUE, painted:  self.theCurrentVolumeSliceImage", CDConstants.DebugTODO )


        # CDConstants.printOut(  "paintTheImageSequence()  --  6.", CDConstants.DebugTODO )
        # TODO remove TODO time.sleep(1.0)


        if ( self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUse3DContours) ) :

            # draw the selected volume slice image, if there is one:
            if isinstance( self.theCurrentContour, QtGui.QImage ) == True:
                lPixMap = QtGui.QPixmap.fromImage(self.theCurrentContour)
                lPainter.drawPixmap(QtCore.QPoint(0,0), lPixMap)
                CDConstants.printOut(  "paintTheImageSequence()  --  6b. --  CDConstants.ImageSequenceUse3DContours TRUE, painted:  self.theCurrentContour", CDConstants.DebugTODO )


        # CDConstants.printOut(  "paintTheImageSequence()  --  7.", CDConstants.DebugTODO )
        # TODO remove TODO time.sleep(1.0)


        if ( self.getAProcessingModeStatusForImageSequenceToPIFF(CDConstants.ImageSequenceUseEdge) ) :

            # draw the selected edge image, if there is one:
            if isinstance( self.theCurrentEdge, QtGui.QImage ) == True:
                lPixMap = QtGui.QPixmap.fromImage(self.theCurrentEdge)
                lPainter.drawPixmap(QtCore.QPoint(0,0), lPixMap)
                CDConstants.printOut(  "paintTheImageSequence()  --  6b. --  CDConstants.ImageSequenceUseEdge TRUE, painted:  self.theCurrentEdge", CDConstants.DebugTODO )



        # CDConstants.printOut(  "paintTheImageSequence()  --  8.", CDConstants.DebugTODO )
        # TODO remove TODO time.sleep(1.0)


        # pop the QPainter's saved state off the stack:
        lPainter.restore()


        # CDConstants.printOut(  "paintTheImageSequence()  --  9.", CDConstants.DebugTODO )
        # TODO remove TODO time.sleep(1.0)


    # end of def paintTheImageSequence(self, pThePainter):
    # ------------------------------------------------------------------




# 
#     # ------------------------------------------------------------------
#     # 2011 - Mitja: this function is NOT to be called directly: it is the callback handler
#     #   for update() and paint() events, and it paints into the passed QPainter parameter
#     # ------------------------------------------------------------------   
#     def paintEvent(self, pThePainter):
# 
#         # one paint cycle has been called:
#         self.repaintEventsCounter = self.repaintEventsCounter + 1
# 
#         CDConstants.printOut("[C] hello, I'm "+str(debugWhoIsTheRunningFunction())+", parent is "+str(debugWhoIsTheParentFunction())+ \
#             " ||||| self.repaintEventsCounter=="+str(self.repaintEventsCounter)+ \
#             " ||||| CDImageSequence.paintTheImageSequence(pThePainter=="+str(pThePainter)+")", CDConstants.DebugTODO )
# 
#         # 2011 - Mitja: call our function doing the actual drawing,
#         #   passing along the QPainter parameter received by paintEvent():
#         self.paintTheImageSequence(pThePainter)



    # ------------------------------------------------------------------
    def getCurrentEdgePixmap(self):
        if isinstance( self.theCurrentEdge, QtGui.QImage ) == True:
            lPixMap = QtGui.QPixmap.fromImage(self.theCurrentEdge)
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: getCurrentEdgePixmap( " \
                + str(self.theCurrentEdge) + " )", CDConstants.DebugExcessive )
            return (  lPixMap  )
        else:
            CDConstants.printOut( "___ - DEBUG ----- CDImageSequence: getCurrentEdgePixmap( " \
                + "NO EDGE IMAGE ) ", CDConstants.DebugExcessive )
            return False




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

#         print "2010 DEBUG:", self.repaintEventsCounter, "CDImageSequence.drawGrid() DONE - WWIIDDTTHH(pifScene) =" ,self.pifSceneWidth, "HHEEIIGGHHTT(pifScene) =",self.pifSceneHeight
       


    # ------------------------------------------------------------------
    def mousePressEvent(self, pEvent):       
        # 2010 - Mitja: track click events of the mouse left button only:
        if pEvent.button() == QtCore.Qt.LeftButton:
            # print "CLICK inside CDImageSequence"

            # this is color-picking mode:
            color = self.theCurrentImage.pixel(pEvent.scenePos().x(),pEvent.scenePos().y())
            self.myMouseX = pEvent.scenePos().x()
            self.myMouseY = pEvent.scenePos().y()
            self.emit(QtCore.SIGNAL("mousePressedInImageLayerSignal()"))


            # 2010 - Mitja: update the CDImageSequence's parent widget,
            #   i.e. paintEvent() will be invoked regardless of the picking mode:
            self.graphicsSceneWindow.scene.update()





    
    
    
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
    
    def qimage2numpy(self, qimage, dtype = 'array'):
        """Convert QImage to numpy.ndarray.  The dtype defaults to uint8
        for QImage.Format_Indexed8 or `bgra_dtype` (i.e. a record array)
        for 32bit color images.  You can pass a different dtype to use, or
        'array' to get a 3D uint8 array for color images."""
        result_shape = (qimage.height(), qimage.width())
        temp_shape = (qimage.height(),
                      qimage.bytesPerLine() * 8 / qimage.depth())
        if qimage.format() in (QtGui.QImage.Format_ARGB32_Premultiplied,
                               QtGui.QImage.Format_ARGB32,
                               QtGui.QImage.Format_RGB32):
            if dtype == 'rec':
                dtype = bgra_dtype
            elif dtype == 'array':
                dtype = numpy.uint8
                result_shape += (4, )
                temp_shape += (4, )
        elif qimage.format() == QtGui.QImage.Format_Indexed8:
            dtype = numpy.uint8
        else:
            raise ValueError("qimage2numpy only supports 32bit and 8bit images")
        # FIXME: raise error if alignment does not match
        buf = qimage.bits().asstring(qimage.numBytes())
        result = numpy.frombuffer(buf, dtype).reshape(temp_shape)
        if result_shape != temp_shape:
            result = result[:,:result_shape[1]]
        if qimage.format() == QtGui.QImage.Format_RGB32 and dtype == numpy.uint8:
            result = result[...,:3]
        return result
    
    def numpy2qimage(self, array):
        if numpy.ndim(array) == 2:
            return self.gray2qimage(array)
        elif numpy.ndim(array) == 3:
            return self.rgb2qimage(array)
        raise ValueError("can only convert 2D or 3D arrays")
    
    def gray2qimage(self, gray):
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
        bgra[...,0] = rgb[...,2]
        bgra[...,1] = rgb[...,1]
        bgra[...,2] = rgb[...,0]
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
    
    
    
    
    





if __name__ == '__main__':

    import sys

    app = QtGui.QApplication(sys.argv)
    window = CDImageSequence()
    window.show()
    window.raise_()
    sys.exit(app.exec_())


# Local Variables:
# coding: US-ASCII
# End:
