#!/usr/bin/env python
#
# CDWaitProgressBarWithImage - add-on QProgressBar and image QWidget for CellDraw - Mitja 2011
#
# ------------------------------------------------------------

import inspect # <-- for debugging functions, may be removed in final version
#
from PyQt4 import QtGui, QtCore
#


# external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# debugging functions, may be removed in final version:
def debugWhoIsTheRunningFunction():
    return inspect.stack()[1][3]
def debugWhoIsTheParentFunction():
    return inspect.stack()[2][3]





# ------------------------------------------------------------
# a class to store a pixmap image, based on PIF_Generator's
#     original code for InputImageLabel()
# ------------------------------------------------------------
class ProgressBarImageLabel(QtGui.QLabel):
    # for unexplained reasons, this class is based on QLabel,
    #   even though it is NOT used as a label.
    #   Instead, this class draws an image, intercepts
    #   mouse click events, etc.

    def __init__(self,parent=None):
        QtGui.QLabel.__init__(self, parent)
        QtGui.QWidget.__init__(self, parent)
        self.width = 10
        self.height = 10
        self.rasterWidth = 10
        # store a pixmap:
        self.setPixmap( QtGui.QPixmap(self.width, self.height) )
        self.pixmap().fill(QtCore.Qt.transparent)

        # the "image" QImage is the one we see, i.e. the rasterized one
        self.image = QtGui.QImage( self.pixmap().toImage() )
        self.x = 0
        self.y = 0
        self.fixedSizeRaster = False
# 
#         # store the pixmap holding the specially rendered scene:
#         self.theRasterizedImageLabel.setPixmap(lPixmap)
#         # this QImage is going to hold the rasterized version:
#         self.theRasterizedImageLabel.image = lPixmap.toImage()
# 
#         self.theRasterizedImageLabel.width = int( lPixmap.width() )
#         self.theRasterizedImageLabel.height = int ( lPixmap.height() )
#         CDConstants.printOut( "___ - DEBUG ----- CDSceneRasterizer: self.setInputGraphicsScene() pGraphicsScene w,h =" + \
#               str(self.theRasterizedImageLabel.width) + " " + str(self.theRasterizedImageLabel.height), CDConstants.DebugVerbose )
# 
#         # adjusts the size of the label widget to fit its contents (i.e. the pixmap):
#         self.theRasterizedImageLabel.adjustSize()
#         self.theRasterizedImageLabel.show()
#         self.theRasterizedImageLabel.update()

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

        if self.fixedSizeRaster == True:
            lPen = QtGui.QPen()
#             lPen.setColor(QtGui.QColor(QtCore.Qt.black))
# TODO TODO: 20111129 TODO: go back to a black grid:
            lTmpRgbaColor = QtGui.QColor( int(random.random()*256.0), \
                                          int(random.random()*256.0), \
                                          int(random.random()*256.0) ).rgba()
            lPen.setColor(QtGui.QColor(lTmpRgbaColor))

            lPen.setWidth(1)
            lPen.setCosmetic(True)
            lPainter.setPen(lPen)
            self.drawGrid(lPainter)
        else:
            # we don't need to draw the grid on top of the label:
            pass

        lPainter.end()

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
#             CDConstants.printOut( " "+str( "___ - DEBUG ----- ProgressBarImageLabel: mousePressEvent() finds Color(x,y) = %s(%s,%s)" %(QtCore.QString("%1").arg(color, 8, 16), self.x, self.y) )+" ", CDConstants.DebugTODO )
#             self.emit(QtCore.SIGNAL("getpos()"))



# ======================================================================
# a QWidget-based progress bar, instead of a dialog-style widget
# ======================================================================
class CDWaitProgressBarWithImage(QtGui.QWidget):

    # ------------------------------------------------------------------
    def __init__(self, pTitle="CellDraw: processing.", pLabelText=" ", pMaxValue=100, pParent=None):
        # it is compulsory to call the parent's __init__ class right away:
        super(CDWaitProgressBarWithImage, self).__init__(pParent)

        # the progress bar widget is defined in createProgressBar() below:
        self.__progressBar = None

        self.__theTitle = pTitle
        self.__theLabelText = pLabelText
        self.__maxValue = pMaxValue
        self.__theParent = pParent

        self.__waitProgressBarGroupBox = self.__InitCentralWidget(self.__theTitle)

        #
        # set up a widget with a QProgressBar and an image, show it inside the QWidget:
        #

        lFont = QtGui.QFont()
        lFont.setWeight(QtGui.QFont.Light)
        self.setFont(lFont)
        self.setLayout(QtGui.QHBoxLayout())
        self.layout().setMargin(0)
        self.layout().setSpacing(0)
        self.layout().setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        self.setWindowTitle("CellDraw: processing windowTitle.")
        self.setStatusTip(QtCore.QString(self.__theTitle))
        self.setToolTip(QtCore.QString(self.__theTitle))
        # do not delete the window widget when the window is closed:
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        self.layout().addWidget(self.__waitProgressBarGroupBox)
        print "TODO TODO TODO: remove         self.theContentWidget = self.__waitProgressBarGroupBox"
        self.theContentWidget = QtGui.QWidget()
#         self.theContentWidget = self.__waitProgressBarGroupBox

        CDConstants.printOut( " "+str( "--- - DEBUG ----- CDWaitProgressBarWithImage: __init__(): done" )+" ", CDConstants.DebugTODO )

    # end of   def __init__()
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    def hide(self):
        print
        print "-----------------------------------------"
        print "  CDWaitProgressBarWithImage.hide() ....."
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # this code was in the INIT section, but we now create/delete the image label
        #   on the fly when showing/hiding the progress bar widget --->
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#         self.__waitProgressBarGroupBox.layout().removeWidget(self.scrollArea)
#         if (self.theProgressBarImageLabel != None):
#             self.theProgressBarImageLabel.destroy(True, True)
#             self.theProgressBarImageLabel = None
#         if (self.scrollArea != None):
#             self.scrollArea.destroy(True, True)
#             self.scrollArea = None
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


        # to have the parent widget (a QStatusBar object) resize properly, remove self from it:
        self.__theParent.removeWidgetFromStatusBar(self)

        print
        print "  self.__theParent.size() =", self.__theParent.size()
#         print "  self.__theParent ==", self.__theParent, "calling: self.__theParent.resize(16,64) "
#         self.__theParent.resize(16,64)
        self.__theParent.update()
        print "  self.__theParent.size() =", self.__theParent.size()
        print "  self.__theParent ==", self.__theParent, "calling: self.__theParent.reformat() "
        self.__theParent.reformat()
        self.__theParent.update()
        print "  self.__theParent.size() =", self.__theParent.size()

#         self.__theParent.adjustSize()
#         self.adjustSize()

        # pass the hide upwards:
        super(CDWaitProgressBarWithImage, self).hide()

        print
        print "  CDWaitProgressBarWithImage.hide() done."
        print "-----------------------------------------"

    # ------------------------------------------------------------------
    def show(self):
        print
        print "-----------------------------------------"
        print "  CDWaitProgressBarWithImage.show() ....."


        # pass the show upwards:
        super(CDWaitProgressBarWithImage, self).show()


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # this code was in the INIT section, but we now create/delete the image label
        #   on the fly when showing/hiding the progress bar widget --->
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 
#         # now create a QLabel, but NOT to be used as a label.
#         #   as in the original PIF_Generator code:
#         self.theProgressBarImageLabel = ProgressBarImageLabel()
# 
#         # set the size policy of the theProgressBarImageLabel widget:.
#         #   "The widget will get as much space as possible."
# #         self.theProgressBarImageLabel.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
# 
#         # according to Qt documentation,
#         #   "scale the pixmap to fill the available space" :
#         self.theProgressBarImageLabel.setScaledContents(True)
# 
#         # self.theProgressBarImageLabel.setLineWidth(1)
#         # self.theProgressBarImageLabel.setMidLineWidth(1)
# 
#         # set a QFrame type for this label, so that it shows up with a visible border around itself:
# #         self.theProgressBarImageLabel.setFrameShape(QtGui.QFrame.Panel)
#         # self.theProgressBarImageLabel.setFrameShadow(QtGui.QFrame.Plain)
# 
#         # self.theProgressBarImageLabel.setAlignment = (QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
#         # self.theProgressBarImageLabel.setObjectName("theProgressBarImageLabel")
# 
#         # self.theProgressBarImageLabel.update()
#         
# 
#         # for unexplained reasons, a QLabel containing an image has to be placed
#         #   in a QScrollArea to be displayed within a layout. Placing theProgressBarImageLabel
#         #   directly in the layout would *not* display it at all!
#         #   Therefore, create a QScrollArea and assign theProgressBarImageLabel to it:
#         self.scrollArea = QtGui.QScrollArea()
#         # self.scrollArea.setBackgroundRole(QtGui.QPalette.AlternateBase)
# #         self.scrollArea.setBackgroundRole(QtGui.QPalette.Mid)
#         self.scrollArea.setWidget(self.theProgressBarImageLabel)
#         self.scrollArea.setAlignment = (QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
#         
#         print "  self.theProgressBarImageLabel =", self.theProgressBarImageLabel
#         print "  self.scrollArea =", self.scrollArea
# 
#         self.__waitProgressBarGroupBox.layout().addWidget(self.scrollArea)
# 
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # <---   this code was in the INIT section, but we now create/delete
        # the image label on the fly when showing/hiding the progress bar widget
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


        # to have the parent widget (a QStatusBar object) resize properly, insert self in it:
        self.__theParent.insertPermanentWidgetInStatusBar(0, self)

        print
        print "  self.__theParent.size() =", self.__theParent.size()
#         print "  self.__theParent ==", self.__theParent, "calling: self.__theParent.resize(16,64) "
#         self.__theParent.resize(16,64)
        self.__theParent.update()
        print "  self.__theParent.size() =", self.__theParent.size()
        print "  self.__theParent ==", self.__theParent, "calling: self.__theParent.reformat() "
        self.__theParent.reformat()
        self.__theParent.update()
        print "  self.__theParent.size() =", self.__theParent.size()



#         self.__theParent.adjustSize()
#         self.adjustSize()


        print
        print "  CDWaitProgressBarWithImage.show() done."
        print "-----------------------------------------"


    # ------------------------------------------------------------------
    # init - central widget containing a QProgressBar and an image, set up and show:
    # ------------------------------------------------------------------
    def __InitCentralWidget(self, pTitle):
        # -------------------------------------------






        # now create a QLabel, but NOT to be used as a label.
        #   as in the original PIF_Generator code:
        self.theProgressBarImageLabel = ProgressBarImageLabel()

        # set the size policy of the theProgressBarImageLabel widget:.
        #   "The widget will get as much space as possible."
#         self.theProgressBarImageLabel.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)

        # according to Qt documentation,
        #   "scale the pixmap to fill the available space" :
        self.theProgressBarImageLabel.setScaledContents(False)

        # self.theProgressBarImageLabel.setLineWidth(1)
        # self.theProgressBarImageLabel.setMidLineWidth(1)

        # set a QFrame type for this label, so that it shows up with a visible border around itself:
#         self.theProgressBarImageLabel.setFrameShape(QtGui.QFrame.Panel)
        # self.theProgressBarImageLabel.setFrameShadow(QtGui.QFrame.Plain)

        # self.theProgressBarImageLabel.setAlignment = (QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        # self.theProgressBarImageLabel.setObjectName("theProgressBarImageLabel")

        # self.theProgressBarImageLabel.update()
        

        # for unexplained reasons, a QLabel containing an image has to be placed
        #   in a QScrollArea to be displayed within a layout. Placing theProgressBarImageLabel
        #   directly in the layout would *not* display it at all!
        #   Therefore, create a QScrollArea and assign theProgressBarImageLabel to it:
        self.scrollArea = QtGui.QScrollArea()
        # self.scrollArea.setBackgroundRole(QtGui.QPalette.AlternateBase)
#         self.scrollArea.setBackgroundRole(QtGui.QPalette.Mid)
        self.scrollArea.setWidget(self.theProgressBarImageLabel)
        self.scrollArea.setAlignment = (QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        
        print "  self.theProgressBarImageLabel =", self.theProgressBarImageLabel
        print "  self.scrollArea =", self.scrollArea







        lGroupBox = QtGui.QGroupBox(pTitle)
#         lGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,0,222)))
#         lGroupBox.setAutoFillBackground(True)
        lGroupBox.setLayout(QtGui.QHBoxLayout())
        lGroupBox.layout().setMargin(0)
        lGroupBox.layout().setSpacing(2)
        lGroupBox.layout().setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        # this self.__infoLabel part is cosmetic and can safely be removed,
        #     unless useful info is provided here:
        self.__infoLabel = QtGui.QLabel()
        self.__infoLabel.setText(self.__theLabelText)
        self.__infoLabel.setAlignment = QtCore.Qt.AlignCenter
#         self.__infoLabel.setLineWidth(3)
#         self.__infoLabel.setMidLineWidth(3)

        # create a progress bar:
        self.createProgressBar()      

        # this self.percentageLabel part is cosmetic and can safely be removed,
        #     unless useful info is provided here:
        self.percentageLabel = QtGui.QLabel()
        self.percentageLabel.setText("0 %")
        self.percentageLabel.setAlignment = QtCore.Qt.AlignCenter
        self.percentageLabel.setLineWidth(3)
        self.percentageLabel.setMidLineWidth(3)

        # place all 'sub-widgets' in the layout:

        lVBoxLayout = QtGui.QVBoxLayout()
        lVBoxLayout.setMargin(0)
        lVBoxLayout.setSpacing(2)
        lVBoxLayout.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        lVBoxLayout.addWidget(self.__infoLabel)
        lVBoxLayout.addWidget(self.__progressBar)
        lVBoxLayout.addWidget(self.percentageLabel)


#         self.theProgressBarImageLabel = None
#         self.scrollArea = None

        lGroupBox.layout().addLayout(lVBoxLayout)
        lGroupBox.layout().addWidget(self.scrollArea)

        # finally place the complete layout in a QWidget and return it:
        return lGroupBox

    # end of   def __InitCentralWidget(self)
    # ------------------------------------------------------------------



    # ---------------------------------------------------------
    def setTitle(self, pCaption):
        self.__theTitle = str(pCaption)
        # self.__infoLabel.setText(str(pCaption))
        self.__waitProgressBarGroupBox.setTitle(self.__theTitle)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    # ---------------------------------------------------------
    def setValue(self, pValue):
        self.__progressBar.setValue(pValue)

        curVal = self.__progressBar.value()
        maxVal = self.__progressBar.maximum()
        lPercentage = (float(curVal) / float(maxVal)) * 100.0
        self.percentageLabel.setText( QtCore.QString("... %1 %").arg(lPercentage, 0, 'g', 2) )
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    # ---------------------------------------------------------
    def setTitleTextRange(self, pCaption="CellDraw: processing.", pLabelText=" ", pMin=0, pMax=100):
        self.__infoLabel.setText(pLabelText)
        if (pLabelText==" "):
            self.__infoLabel.hide()
        else:
            self.__infoLabel.show()
        self.__progressBar.setRange(pMin, pMax)
        self.__progressBar.setValue(pMin)
        self.__theTitle = pCaption
        self.__waitProgressBarGroupBox.setTitle(self.__theTitle)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    # ---------------------------------------------------------
    def setInfoText(self, pLabelText=" "):
        self.__infoLabel.setText(pLabelText)
        if (pLabelText==" "):
            self.__infoLabel.hide()
        else:
            self.__infoLabel.show()
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    # ---------------------------------------------------------
    def setRange(self, pMin=0, pMax=100):
        self.__progressBar.setRange(pMin, pMax)
        self.__progressBar.setValue(pMin)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    # ---------------------------------------------------------
    def createProgressBar(self):
        self.__progressBar = QtGui.QProgressBar()
        # self.__progressBar.setRange(0, 10000)
        self.__progressBar.setRange(0, self.__maxValue)
        self.__progressBar.setValue(0)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    # ---------------------------------------------------------
    def advanceProgressBar(self):
        curVal = self.__progressBar.value()
        maxVal = self.__progressBar.maximum()
        # self.__progressBar.setValue(curVal + (maxVal - curVal) / 100)
        lPercentage = (float(curVal) / float(maxVal)) * 100.0
        # CDConstants.printOut( " "+str( "ah yes", curVal, maxVal, lPercentage, QtCore.QString("%1").arg(lPercentage) )+" ", CDConstants.DebugTODO )
        self.percentageLabel.setText( QtCore.QString("... %1 %").arg(lPercentage, 0, 'g', 2) )
        self.__progressBar.setValue(curVal + 1)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
               
    # ---------------------------------------------------------
    def resetProgressBar(self):
        self.percentageLabel.setText("0 %")
        self.__progressBar.setValue(0)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    # ---------------------------------------------------------
    def maxProgressBar(self):
        self.percentageLabel.setText("100 %")
        self.__progressBar.setValue(self.__maxValue)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
    # ---------------------------------------------------------




# ======================================================================
# the following if statement checks whether the present file
#    is currently being used as standalone (main) program, and in this
#    class's (CDWaitProgressBarWithImage) case it is simply used for ***testing***:
# ======================================================================
if __name__ == '__main__':

    import sys     # <-- for command-line arguments, may be removed in final version

    CDConstants.printOut( "__main__() running:"+str( debugWhoIsTheRunningFunction() ), CDConstants.DebugTODO )
    # CDConstants.printOut( " "+str( "parent:",  debugWhoIsTheParentFunction() )+" ", CDConstants.DebugTODO )

    # every PyQt4 app must create an application object, from the QtGui module:
    miApp = QtGui.QApplication(sys.argv)

    # the window containing the progress bar:
    mainDialog = CDWaitProgressBarWithImage("some explanatory text", 345)

    # show() and raise_() have to be called here:
    mainDialog.show()
    # raise_() is a necessary workaround to a PyQt-caused (or Qt-caused?) bug on Mac OS X:
    #   unless raise_() is called right after show(), the window/dialog/etc will NOT become
    #   the foreground window and won't receive user input focus:
    mainDialog.raise_()

    sys.exit(miApp.exec_())

# end if __name__ == '__main__'
# Local Variables:
# coding: US-ASCII
# End:
