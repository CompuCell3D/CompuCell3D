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
        self.__width = 64
        self.__height = 64
        self.__rasterWidth = 10
        self.__fixedSizeRaster = False

        # store a pixmap:
        self.setPixmap( QtGui.QPixmap(self.__width, self.__height) )
        self.pixmap().fill(QtCore.Qt.darkGreen)

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

        if self.__fixedSizeRaster == True:
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
            self.__drawGrid(lPainter)
        else:
            # we don't need to draw the grid on top of the label:
            pass

        lPainter.end()

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


    def __drawPixmapAtPoint(self, pPixmap, pXmin=0, pYmin=0):

        lPainter = QtGui.QPainter()
        lPainter.begin(self.pixmap())
        lPainter.drawPixmap(pXmin, pYmin, pPixmap)
        lPainter.end()
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
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # this code was in the INIT section, but we now create/delete the image label
        #   on the fly when showing/hiding the progress bar widget --->
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # to have the parent widget (a QStatusBar object) resize properly, remove self from it:
        if isinstance( self.__theParent, QtGui.QStatusBar ) == True:
            self.__theParent.removeWidgetFromStatusBar(self)

        # finally pass the hide() call upwards:
        super(CDWaitProgressBar, self).hide()

    # end of   def hide(self)
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    def show(self):
        # immediately pass the show() call upwards:
        super(CDWaitProgressBar, self).show()

        # to have the parent widget (a QStatusBar object) resize properly, insert self in it:
        if isinstance( self.__theParent, QtGui.QStatusBar ) == True:
            self.__theParent.insertPermanentWidgetInStatusBar(0, self)

    # end of     def show(self)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # init - central widget containing a QProgressBar, set up and show:
    # ------------------------------------------------------------------
    def __InitCentralWidget(self, pTitle):
        # -------------------------------------------

        # if we needed an image in the progress bar, we'd now:
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

        QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)
        lCriticalErrorWarning = QtGui.QMessageBox.critical( self, \
        "CellDraw", \
        "Critical eror: CDWaitProgressBar.setImagePixmap()\n\n." + \
        "Please contact your system administrator or the source where you obtained CellDraw." )
        sys.exit()

    # end of   def setImagePixmap(self, pPixmap, pWidth=-1, pHeight=-1)
    # ---------------------------------------------------------



    # ---------------------------------------------------------
    def setTitle(self, pCaption):
        self.__theTitle = str(pCaption)
        # self.__infoLabel.setText(str(pCaption))
        self.__waitProgressBarGroupBox.setTitle(self.__theTitle)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)

    # ---------------------------------------------------------
    def setValue(self, pValue):
        self.__progressBar.setValue(pValue)

        curVal = self.__progressBar.value()
        maxVal = self.__progressBar.maximum()
        if (maxVal != 0) :
            lPercentage = (float(curVal) / float(maxVal)) * 100.0
        else:
            lPercentage = 100.0
        self.__percentageLabel.setText( QtCore.QString("... %1 %").arg(lPercentage, 0, 'g', 2) )
        QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)

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
        QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)

    # ---------------------------------------------------------
    def setInfoText(self, pLabelText=" "):
        self.__infoLabel.setText(pLabelText)
        if (pLabelText==" "):
            self.__infoLabel.hide()
        else:
            self.__infoLabel.show()
        QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)

    # ---------------------------------------------------------
    def setRange(self, pMin=0, pMax=100):
        self.__progressBar.setRange(pMin, pMax)
        self.__progressBar.setValue(pMin)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)

    # ---------------------------------------------------------
    def __createProgressBar(self):
        lProgressBar = QtGui.QProgressBar()
        # lProgressBar.setRange(0, 10000)
        lProgressBar.setRange(0, self.__maxValue)
        lProgressBar.setValue(0)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)
        return lProgressBar

    # ---------------------------------------------------------
    def advanceProgressBar(self):
        curVal = self.__progressBar.value()
        maxVal = self.__progressBar.maximum()
        # self.__progressBar.setValue(curVal + (maxVal - curVal) / 100)
        if (maxVal != 0) :
            lPercentage = (float(curVal) / float(maxVal)) * 100.0
        else:
            lPercentage = 100.0
        # CDConstants.printOut( " "+str( "ah yes", curVal, maxVal, lPercentage, QtCore.QString("%1").arg(lPercentage) )+" ", CDConstants.DebugTODO )
        self.__percentageLabel.setText( QtCore.QString("... %1 %").arg(lPercentage, 0, 'g', 2) )
        self.__progressBar.setValue(curVal + 1)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)
               
    # ---------------------------------------------------------
    def resetProgressBar(self):
        self.__percentageLabel.setText("0 %")
        self.__progressBar.setValue(0)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)

    # ---------------------------------------------------------
    def maxProgressBar(self):
        self.__percentageLabel.setText("100 %")
        self.__progressBar.setValue(self.__maxValue)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.AllEvents)
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
