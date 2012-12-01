#!/usr/bin/env python
#
# CDWaitProgressBar - add-on QProgressBar and QWidget for CellDraw - Mitja 2011
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



# ======================================================================
# a QWidget-based progress bar, instead of a dialog-style widget
# ======================================================================
class CDWaitProgressBar(QtGui.QWidget):

    # ------------------------------------------------------------------
    def __init__(self, pTitle="CellDraw: processing.", pLabelText=" ", pMaxValue=100, pParent=None):
        # it is compulsory to call the parent's __init__ class right away:
        super(CDWaitProgressBar, self).__init__(pParent)

        # the progress bar widget is defined in createProgressBar() below:
        self.__progressBar = None

        self.__theTitle = pTitle
        self.__theLabelText = pLabelText
        self.__maxValue = pMaxValue
        self.__theParent = pParent

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
        self.layout().setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        self.setWindowTitle("CellDraw: processing windowTitle.")
        self.setStatusTip(QtCore.QString(self.__theTitle))
        self.setToolTip(QtCore.QString(self.__theTitle))
        # do not delete the window widget when the window is closed:
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        self.layout().addWidget(self.__waitProgressBarGroupBox)

        CDConstants.printOut( " "+str( "--- - DEBUG ----- CDWaitProgressBar: __init__(): done" )+" ", CDConstants.DebugTODO )

    # end of   def __init__()
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    def hide(self):
        print
        print "--------------------------------"
        print "  CDWaitProgressBar.hide() ....."
        # pass the hide upwards:
        super(CDWaitProgressBar, self).hide()

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
        print
        print "  CDWaitProgressBar.hide() done."
        print "--------------------------------"

    # ------------------------------------------------------------------
    def show(self):
        print
        print "--------------------------------"
        print "  CDWaitProgressBar.show() ....."
        # pass the show upwards:
        super(CDWaitProgressBar, self).show()

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
        print
        print "  CDWaitProgressBar.show() done."
        print "--------------------------------"


    # ------------------------------------------------------------------
    # init - central widget containing a QProgressBar, set up and show:
    # ------------------------------------------------------------------
    def __InitCentralWidget(self, pTitle):
        # -------------------------------------------

        lGroupBox = QtGui.QGroupBox(pTitle)
#         lGroupBox.setPalette(QtGui.QPalette(QtGui.QColor(222,0,222)))
#         lGroupBox.setAutoFillBackground(True)
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
        self.createProgressBar()      

        # this self.percentageLabel part is cosmetic and can safely be removed,
        #     unless useful info is provided here:
        self.percentageLabel = QtGui.QLabel()
        self.percentageLabel.setText("0 %")
        self.percentageLabel.setAlignment = QtCore.Qt.AlignCenter
        self.percentageLabel.setLineWidth(3)
        self.percentageLabel.setMidLineWidth(3)

        # place all 'sub-widgets' in the layout:
        lGroupBox.layout().addWidget(self.__infoLabel)
        lGroupBox.layout().addWidget(self.__progressBar)
        lGroupBox.layout().addWidget(self.percentageLabel)

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
#    class's (CDWaitProgressBar) case it is simply used for ***testing***:
# ======================================================================
if __name__ == '__main__':

    import sys     # <-- for command-line arguments, may be removed in final version

    CDConstants.printOut( "__main__() running:"+str( debugWhoIsTheRunningFunction() ), CDConstants.DebugTODO )
    # CDConstants.printOut( " "+str( "parent:",  debugWhoIsTheParentFunction() )+" ", CDConstants.DebugTODO )

    # every PyQt4 app must create an application object, from the QtGui module:
    miApp = QtGui.QApplication(sys.argv)

    # the window containing the progress bar:
    mainDialog = CDWaitProgressBar("some explanatory text", 345)

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
