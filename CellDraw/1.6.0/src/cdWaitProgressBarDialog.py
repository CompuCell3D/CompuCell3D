#!/usr/bin/env python
#
# CDWaitProgressBarDialog - add-on QProgressBar dialog for CellDraw - Mitja 2011
#
# ------------------------------------------------------------

import inspect # <-- for debugging functions, may be removed in final version
#
from PyQt4 import QtGui, QtCore
#


# debugging functions, may be removed in final version:
def debugWhoIsTheRunningFunction():
    return inspect.stack()[1][3]
def debugWhoIsTheParentFunction():
    return inspect.stack()[2][3]



# ======================================================================
# a QDialog-based widget dialog, in application-specific dialog style
# ======================================================================
class CDWaitProgressBarDialog(QtGui.QDialog):

    def __init__(self, pTitle="CellDraw: processing.", pMaxValue=100, pParent=None):
        # it is compulsory to call the parent's __init__ class right away:
        super(CDWaitProgressBarDialog, self).__init__(pParent)

        # delete the window widget when its window is closed:
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
       
        self.theTitle = pTitle
        self.maxValue = pMaxValue
        self.theParent = pParent

        #
        # init (1) - windowing GUI stuff:
        #
        self.miInitGUI()

        #
        # init (2) - set up a widget with a QProgressBar, show it inside the dialog:
        #
        self.layout().addWidget(self.miInitCentralWidget())
        self.layout().setMargin(2)

        # CDConstants.printOut( " "+str( "--- - DEBUG ----- CDWaitProgressBarDialog: __init__(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------------
    # init (1) - windowing GUI stuff:
    # ------------------------------------------------------------------
    def miInitGUI(self):

        # this is a progress dialog within a processing operation, set it as modal:
        self.setModal(True)

        # how will the CDWaitProgressBarDialog look like:
        self.setWindowTitle("CellDraw: processing.")
        self.setMinimumWidth(320)
        # setGeometry is inherited from QWidget, taking 4 arguments:
        #   x,y  of the top-left corner of the QWidget, from top-left of screen
        #   w,h  of the QWidget
        # NOTE: the x,y is NOT the top-left edge of the window,
        #    but of its **content** (excluding the menu bar, toolbar, etc.
        # self.setGeometry(750,480,480,320)

        # QVBoxLayout layout lines up widgets vertically:
        self.setLayout(QtGui.QVBoxLayout())
        self.layout().setAlignment(QtCore.Qt.AlignTop)
        self.layout().setMargin(2)

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
        #    so we use a plain QtCore.Qt.Dialog type instead:
        miDialogsWindowFlags = QtCore.Qt.Dialog
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
    # init (2) - central widget containing a QProgressBar, set up and show:
    # ------------------------------------------------------------------
    def miInitCentralWidget(self):
        # -------------------------------------------
        # the dialog's vbox layout
        theContainerWidget = QtGui.QWidget()

        # this infoLabel part is cosmetic and can safely be removed,
        #     unless useful info is provided here:
        infoLabel = QtGui.QLabel()
        infoLabel.setText(self.theTitle)
        infoLabel.setAlignment = QtCore.Qt.AlignCenter
        infoLabel.setLineWidth(3)
        infoLabel.setMidLineWidth(3)

        # create a progress bar:
        self.createProgressBar()      

        # this self.percentageLabel part is cosmetic and can safely be removed,
        #     unless useful info is provided here:
        self.percentageLabel = QtGui.QLabel()
        self.percentageLabel.setText("0 %")
        self.percentageLabel.setAlignment = QtCore.Qt.AlignCenter
        self.percentageLabel.setLineWidth(3)
        self.percentageLabel.setMidLineWidth(3)

        # create a layout and place all 'sub-widgets' in it:
        vbox = QtGui.QVBoxLayout()
        vbox.setMargin(2)
        vbox.addWidget(infoLabel)
        vbox.addWidget(self.progressBar)
        vbox.addWidget(self.percentageLabel)

        # finally place the complete layout in a QWidget and return it:
        theContainerWidget.setLayout(vbox)
        return theContainerWidget




    # ---------------------------------------------------------
    def setValue(self, pValue):
        self.progressBar.setValue(pValue)

        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        lPercentage = (float(curVal) / float(maxVal)) * 100.0
        self.percentageLabel.setText( QtCore.QString("... %1 %").arg(lPercentage, 0, 'g', 2) )
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    # ---------------------------------------------------------
    def setRange(self, pMin=0, pMax=100):
        self.progressBar.setRange(pMin, pMax)
        self.progressBar.setValue(pMin)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    # ---------------------------------------------------------
    def createProgressBar(self):
        self.progressBar = QtGui.QProgressBar()
        # self.progressBar.setRange(0, 10000)
        self.progressBar.setRange(0, self.maxValue)
        self.progressBar.setValue(0)
    # ---------------------------------------------------------
    def advanceProgressBar(self):
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        # self.progressBar.setValue(curVal + (maxVal - curVal) / 100)
        lPercentage = (float(curVal) / float(maxVal)) * 100.0
        # CDConstants.printOut( " "+str( "ah yes", curVal, maxVal, lPercentage, QtCore.QString("%1").arg(lPercentage) )+" ", CDConstants.DebugTODO )
        self.percentageLabel.setText( QtCore.QString("... %1 %").arg(lPercentage, 0, 'g', 2) )
        self.progressBar.setValue(curVal + 1)
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
               
    # ---------------------------------------------------------
    def resetProgressBar(self):
        self.percentageLabel.setText("0 %")
        self.progressBar.setValue(0)
    # ---------------------------------------------------------
    def maxProgressBar(self):
        self.percentageLabel.setText("100 %")
        self.progressBar.setValue(self.maxValue)
    # ---------------------------------------------------------




# ======================================================================
# the following if statement checks whether the present file
#    is currently being used as standalone (main) program, and in this
#    class's (CDWaitProgressBarDialog) case it is simply used for ***testing***:
# ======================================================================
if __name__ == '__main__':

    import sys     # <-- for command-line arguments, may be removed in final version

    CDConstants.printOut( "__main__() running:"+str( debugWhoIsTheRunningFunction() ), CDConstants.DebugTODO )
    # CDConstants.printOut( " "+str( "parent:",  debugWhoIsTheParentFunction() )+" ", CDConstants.DebugTODO )

    # every PyQt4 app must create an application object, from the QtGui module:
    miApp = QtGui.QApplication(sys.argv)

    # the window containing the progress bar:
    mainDialog = CDWaitProgressBarDialog("some explanatory text", 345)

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
