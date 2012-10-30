#!/usr/bin/env python
#
# CDWaitProgressBarWithImage - add-on QProgressBar dialog for CellDraw - Mitja 2011
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





# ------------------------------------------------------------
# a class to store a pixmap image, based on PIF_Generator's
#     original code for InputImageLabel()
# ------------------------------------------------------------
class ProgressBarImageLabel(QtGui.QLabel):
    # 2010 - Mitja: for unexplained reasons, this class is based on QLabel,
    #   even though it is NOT used as a label.
    #   Instead, this class draws an image, intercepts
    #   mouse click events, etc.

    def __init__(self,parent=None):
        QtGui.QLabel.__init__(self, parent)
        QtGui.QWidget.__init__(self, parent)
        self.width = 240
        self.height = 180
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
# a QDialog-based widget dialog, in application-specific dialog style
# ======================================================================
class CDWaitProgressBarWithImage(QtGui.QDialog):

    def __init__(self, pTitle="CellDraw: processing.", pMaxValue=100, pParent=None):
        # it is compulsory to call the parent's __init__ class right away:
        super(CDWaitProgressBarWithImage, self).__init__(pParent)

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
        self.theContentWidget = self.miInitCentralWidget()
        self.layout().addWidget(self.theContentWidget)
        self.layout().setMargin(2)

        # CDConstants.printOut( " "+str( "--- - DEBUG ----- CDWaitProgressBarWithImage: __init__(): done" )+" ", CDConstants.DebugTODO )


    # ------------------------------------------------------------------
    # init (1) - windowing GUI stuff:
    # ------------------------------------------------------------------
    def miInitGUI(self):

        # this is a progress dialog within a processing operation, set it as modal:
        self.setModal(True)

        # how will the CDWaitProgressBarWithImage look like:
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

        # self.theTitleLabel part is cosmetic and can safely be removed,
        #     unless useful info is provided here:
        self.theTitleLabel = QtGui.QLabel()
        self.theTitleLabel.setText(self.theTitle)
        self.theTitleLabel.setAlignment = QtCore.Qt.AlignCenter
        self.theTitleLabel.setLineWidth(3)
        self.theTitleLabel.setMidLineWidth(3)

        # create a progress bar:
        self.createProgressBar()      

        # this self.percentageLabel part is cosmetic and can safely be removed,
        #     unless useful info is provided here:
        self.percentageLabel = QtGui.QLabel()
        self.percentageLabel.setText("0 %")
        self.percentageLabel.setAlignment = QtCore.Qt.AlignCenter
        self.percentageLabel.setLineWidth(3)
        self.percentageLabel.setMidLineWidth(3)





        # 2010 - Mitja: now create a QLabel, but NOT to be used as a label.
        #   as in the original PIF_Generator code:
        self.theProgressBarImageLabel = ProgressBarImageLabel()

        # 2010 - Mitja: set the size policy of the theProgressBarImageLabel widget:.
        #   "The widget will get as much space as possible."
        self.theProgressBarImageLabel.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)

        # 2010 - Mitja: according to Qt documentation,
        #   "scale the pixmap to fill the available space" :
        self.theProgressBarImageLabel.setScaledContents(False)

        # self.theProgressBarImageLabel.setLineWidth(1)
        # self.theProgressBarImageLabel.setMidLineWidth(1)

        # set a QFrame type for this label, so that it shows up with a visible border around itself:
        self.theProgressBarImageLabel.setFrameShape(QtGui.QFrame.Panel)
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
        self.scrollArea.setBackgroundRole(QtGui.QPalette.Mid)
        self.scrollArea.setWidget(self.theProgressBarImageLabel)
        self.scrollArea.setAlignment = (QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        # create a layout and place all 'sub-widgets' in it:
        vbox = QtGui.QVBoxLayout()
        vbox.setMargin(2)
        vbox.addWidget(self.theTitleLabel)
        vbox.addWidget(self.progressBar)
        vbox.addWidget(self.percentageLabel)
        vbox.addWidget(self.scrollArea)

        # finally place the complete layout in a QWidget and return it:
        theContainerWidget.setLayout(vbox)
        return theContainerWidget



    # ---------------------------------------------------------
    def setTitle(self, pValue):
        self.theTitleLabel.setText(str(pValue))
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)


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
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

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
        QtGui.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    # ---------------------------------------------------------
    def maxProgressBar(self):
        self.percentageLabel.setText("100 %")
        self.progressBar.setValue(self.maxValue)
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
