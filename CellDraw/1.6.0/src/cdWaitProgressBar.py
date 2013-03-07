#!/usr/bin/env python
#
# CDWaitProgressBar - add-on QProgressBar and QWidget for CellDraw - Mitja 2011
#
# ------------------------------------------------------------

import inspect # <-- for debugging functions, may be removed in final version
#
from PyQt4 import QtGui, QtCore
#

import time    # for sleep()


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
#         QtGui.QWidget.__init__(self, parent)
        self.__width = 64
        self.__height = 64
        self.__rasterWidth = 10
        # store a pixmap:
        self.setPixmap( QtGui.QPixmap(self.__width, self.__height) )
        self.pixmap().fill(QtCore.Qt.darkGreen)

        self.__fixedSizeRaster = False

    def paintEvent(self, event):

        # QtGui.QLabel.paintEvent(self,event)

        # start a QPainter on this QLabel - this is why we pass "self" as paramter:
        __lPainter = QtGui.QPainter(self)

        # take care of the RHS <-> LHS mismatch at its visible end,
        #   by flipping the y coordinate in the QPainter's affine transformations:       
        __lPainter.translate(0.0, float(self.pixmap().height()))
        __lPainter.scale(1.0, -1.0)

        # access the QLabel's pixmap to draw it explicitly, using QPainter's scaling:
        __lPainter.drawPixmap(0, 0, self.pixmap())

        if self.__fixedSizeRaster == True:
            __lPen = QtGui.QPen()
#             __lPen.setColor(QtGui.QColor(QtCore.Qt.black))
# TODO TODO: 20111129 TODO: go back to a black grid:
            lTmpRgbaColor = QtGui.QColor( int(random.random()*256.0), \
                                          int(random.random()*256.0), \
                                          int(random.random()*256.0) ).rgba()
            __lPen.setColor(QtGui.QColor(lTmpRgbaColor))

            __lPen.setWidth(1)
            __lPen.setCosmetic(True)
            __lPainter.setPen(__lPen)
            self.__drawGrid(__lPainter)
        else:
            # we don't need to draw the grid on top of the label:
            pass

        __lPainter.end()
        # JUJU LA PRIMA VOLTA QUA APPARI EL PROGRESSBAR NEL TOOLBAR DE SOTO
        # JUJU LA SECONDA VOLTA QUA SE RIDIMENSIONA LIMMAGINE NEL PROGRESSBAR A 100x100 pixel

    def __drawGrid(self,painter):
        for x in xrange(0, self.__width, self.__rasterWidth):
            #draw.line([(x, 0), (x, h)], width=2, fill='#000000')
            painter.drawLine(x,0,x,self.__height)
        for y in xrange(0, self.__height, self.__rasterWidth):
         #draw.line([(0, y), (w, y)], width=2, fill='#000000')
            painter.drawLine(0,y,self.__width,y)


    def __plotRect(self, pRGBA, pXmin, pYmin, pXmax, pYmax):

        lColor = QtGui.QColor()
        lColor.setRgba(pRGBA)

        __lPen = QtGui.QPen()
        __lPen.setColor(lColor)
        __lPen.setWidth(1)
        __lPen.setCosmetic(True)

        __lPainter = QtGui.QPainter()
        __lPainter.begin(self.pixmap())
        __lPainter.setPen(__lPen)

        if (pXmin >= pXmax) or (pYmin >= pYmax) :
            # if passed an incorrect rectangle (with max point < min point)
            # then just draw a 3x3 square around the min point
            __lPainter.drawRect(pXmin-1, pYmin-1, 3, 3)

        else:
            __lPainter.drawRect(pXmin, pYmin, (pXmax-pXmin), (pYmax-pYmin))
   
#             __lPen.setColor(QtGui.QColor(QtCore.Qt.black))
#             __lPen.setWidth(1)
#             __lPen.setCosmetic(True)
#    
#             __lPainter.setPen(__lPen)
#             __lPainter.drawRect(pXmin-1, pYmin-1, 3, 3)

        __lPainter.end()


    def __drawPixmapAtPoint(self, pPixmap, pXmin=0, pYmin=0):

        __lPainter = QtGui.QPainter()
        __lPainter.begin(self.pixmap())
        __lPainter.drawPixmap(pXmin, pYmin, pPixmap)
        __lPainter.end()
        self.update()


    def __drawFixedSizeRaster(self, pFixedOrNot=False):
        self.__fixedSizeRaster = pFixedOrNot
        self.update()




# ======================================================================
# a QWidget-based progress bar, instead of a dialog-style widget
# ======================================================================
class CDWaitProgressBar(QtGui.QWidget):

    # ------------------------------------------------------------------
    def __init__(self, pTitle="CellDraw: processing.", pLabelText=" ", pMaxValue=100, pParent=None):
        # it is compulsory to call the parent's __init__ class right away:
        super(CDWaitProgressBar, self).__init__(pParent)


#         print "CDWaitProgressBar.__init__()"


        # the progress bar widget is defined in __createProgressBar() below:
        self.__progressBar = None

        self.__theTitle = pTitle
        self.__theLabelText = pLabelText
        self.__maxValue = pMaxValue
        self.__theParent = pParent

        # if we needed an image in the progress bar, we'd now:
        # self.__imageLabel, self.__waitProgressBarGroupBox = self.__InitCentralWidget(self.__theTitle)
        self.__waitProgressBarGroupBox = self.__InitCentralWidget(self.__theTitle)

        #
        # set up a widget with a QProgressBar, show it inside the QWidget:
        #

        lFont = QtGui.QFont()
        lFont.setWeight(QtGui.QFont.Light)
        self.setFont(lFont)
        self.setLayout(QtGui.QHBoxLayout())
        self.layout().setMargin(0)
        self.layout().setSpacing(0)
        self.layout().setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.setWindowTitle("CellDraw: processing windowTitle.")
        self.setStatusTip(QtCore.QString(self.__theTitle))
        self.setToolTip(QtCore.QString(self.__theTitle))
        # do not delete the window widget when the window is closed:
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        # if we needed an image in the progress bar, we'd now:
#         self.layout().addWidget(self.__imageLabel)

        self.layout().addWidget(self.__waitProgressBarGroupBox)

        CDConstants.printOut( " "+str( "--- - DEBUG ----- CDWaitProgressBar: __init__(): done" )+" ", CDConstants.DebugTODO )

    # end of   def __init__()
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    def hide(self):
#         print
#         print "--------------------------------"
#         print "  CDWaitProgressBar.hide() ....."
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # this code was in the INIT section, but we now create/delete the image label
        #   on the fly when showing/hiding the progress bar widget --->
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # to have the parent widget (a QStatusBar object) resize properly, remove self from it:
        if isinstance( self.__theParent, QtGui.QStatusBar ) == True:
            self.__theParent.removeWidgetFromStatusBar(self)

#             print
#             print "  self.__theParent.size() =", self.__theParent.size()
#     #         print "  self.__theParent ==", self.__theParent, "calling: self.__theParent.resize(16,64) "
#     #         self.__theParent.resize(16,64)
#             self.__theParent.update()
#             print "  self.__theParent.size() =", self.__theParent.size()
#             print "  self.__theParent ==", self.__theParent, "calling: self.__theParent.reformat() "
#             self.__theParent.reformat()
#             self.__theParent.update()
#             print "  self.__theParent.size() =", self.__theParent.size()


        # finally pass the hide() call upwards:
        super(CDWaitProgressBar, self).hide()


#         print
#         print "  CDWaitProgressBar.hide() done."
#         print "--------------------------------"
    # end of   def hide(self)
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    def show(self):
#         print
#         print "--------------------------------"
#         print "  CDWaitProgressBar.show() ....."

        # immediately pass the show() call upwards:
        super(CDWaitProgressBar, self).show()

        # to have the parent widget (a QStatusBar object) resize properly, insert self in it:
        if isinstance( self.__theParent, QtGui.QStatusBar ) == True:
            self.__theParent.insertPermanentWidgetInStatusBar(0, self)

#             print
#             print "  self.__theParent.size() =", self.__theParent.size()
#     #         print "  self.__theParent ==", self.__theParent, "calling: self.__theParent.resize(16,64) "
#     #         self.__theParent.resize(16,64)
#             self.__theParent.update()
#             print "  self.__theParent.size() =", self.__theParent.size()
#             print "  self.__theParent ==", self.__theParent, "calling: self.__theParent.reformat() "
#             self.__theParent.reformat()
#             self.__theParent.update()
#             print "  self.__theParent.size() =", self.__theParent.size()

#         print
#         print "  CDWaitProgressBar.show() done."
#         print "--------------------------------"
    # end of     def show(self)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # init - central widget containing a QProgressBar, set up and show:
    # ------------------------------------------------------------------
    def __InitCentralWidget(self, pTitle):
        # -------------------------------------------

        # if we needed an image in the progress bar, we'd now:
        #   create a QLabel, NOT to be used as a label but to show an image
        #   as in the original PIF_Generator code:
#         self.__theProgressBarImageLabel = ProgressBarImageLabel()
#         print "CDWaitProgressBar.__InitCentralWidget()  self.__theProgressBarImageLabel =="+str(self.__theProgressBarImageLabel)

        lGroupBox = QtGui.QGroupBox(pTitle)
        lGroupBox.setLayout(QtGui.QVBoxLayout())
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
        self.__progressBar = self.__createProgressBar()      

        # this self.__percentageLabel part is cosmetic and can safely be removed,
        #     unless useful info is provided here:
        self.__percentageLabel = QtGui.QLabel()
        self.__percentageLabel.setText("0 %")
        self.__percentageLabel.setAlignment = QtCore.Qt.AlignCenter
        self.__percentageLabel.setLineWidth(3)
        self.__percentageLabel.setMidLineWidth(3)

        # place all 'sub-widgets' in the layout:
        lGroupBox.layout().addWidget(self.__infoLabel)
        lGroupBox.layout().addWidget(self.__progressBar)
        lGroupBox.layout().addWidget(self.__percentageLabel)

        # finally place the complete layout in a QWidget and return it:
#        return lGroupBox

#         print "CDWaitProgressBar.__InitCentralWidget()  self.__theProgressBarImageLabel =="+str(self.__theProgressBarImageLabel)+ " lGroupBox =="+str(lGroupBox)
        # if we needed an image in the progress bar, we'd now:
#         return self.__theProgressBarImageLabel, lGroupBox
        return lGroupBox

    # end of   def __InitCentralWidget(self)
    # ------------------------------------------------------------------




    # ---------------------------------------------------------
    def setImagePixmap(self, pPixmap, pWidth=-1, pHeight=-1):

#         print "CDWaitProgressBar.setImagePixmap() - start.  pPixmap="+str(pPixmap)+", pWidth="+str(pWidth)+", pHeight="+str(pHeight)+" ..."
#         QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
# 
#         print "CDWaitProgressBar.setImagePixmap() - doing self.__theProgressBarImageLabel.setPixmap(pPixmap):"


        # if we needed an image in the progress bar, we'd now:
#         if isinstance( pPixmap, QtGui.QPixmap ) == True:
#             self.__theProgressBarImageLabel.setPixmap(pPixmap)
#         else:
#             # store a dummy pixmap:
#             self.__theProgressBarImageLabel.setPixmap( QtGui.QPixmap(64, 64) )
#             self.__theProgressBarImageLabel.pixmap().fill(QtCore.Qt.darkGreen)

#         print "CDWaitProgressBar.setImagePixmap() - doing nothing."
        # time.sleep(3.0)


        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

#         print "CDWaitProgressBar.setImagePixmap() - end."

    # end of   def setImagePixmap(self, pPixmap, pWidth=-1, pHeight=-1)
    # ---------------------------------------------------------



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
        self.__percentageLabel.setText( QtCore.QString("... %1 %").arg(lPercentage, 0, 'g', 2) )
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
    def __createProgressBar(self):
        lProgressBar = QtGui.QProgressBar()
        # lProgressBar.setRange(0, 10000)
        lProgressBar.setRange(0, self.__maxValue)
        lProgressBar.setValue(0)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
        return lProgressBar

    # ---------------------------------------------------------
    def advanceProgressBar(self):
        curVal = self.__progressBar.value()
        maxVal = self.__progressBar.maximum()
        # self.__progressBar.setValue(curVal + (maxVal - curVal) / 100)
        lPercentage = (float(curVal) / float(maxVal)) * 100.0
        # CDConstants.printOut( " "+str( "ah yes", curVal, maxVal, lPercentage, QtCore.QString("%1").arg(lPercentage) )+" ", CDConstants.DebugTODO )
        self.__percentageLabel.setText( QtCore.QString("... %1 %").arg(lPercentage, 0, 'g', 2) )
        self.__progressBar.setValue(curVal + 1)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
               
    # ---------------------------------------------------------
    def resetProgressBar(self):
        self.__percentageLabel.setText("0 %")
        self.__progressBar.setValue(0)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    # ---------------------------------------------------------
    def maxProgressBar(self):
        self.__percentageLabel.setText("100 %")
        self.__progressBar.setValue(self.__maxValue)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
    # ---------------------------------------------------------




# ======================================================================
# the following if statement checks whether the present file
#    is currently being used as standalone (main) program, and in this
#    class's (CDWaitProgressBar) case it is simply used for ***testing***:
# ======================================================================
if __name__ == '__main__':

    import sys     # <-- for command-line arguments, may be removed in final version

    CDConstants.printOut( "__main__() running:"+str( debugWhoIsTheRunningFunction() ), CDConstants.DebugTODO )
    # CDConstants.printOut( " "+str( "parent:",  debugWhoIsTheParentFunction() )+" ", CDConstants.DebugTODO )

    # every PyQt4 app must create an application object, from the QtGui module:
    miApp = QtGui.QApplication(sys.argv)

    # the window containing the progress bar:
    mainDialog = CDWaitProgressBar("Some text.", "Some more text.", 345)

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
