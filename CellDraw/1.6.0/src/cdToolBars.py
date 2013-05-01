#!/usr/bin/env python

from PyQt4 import QtGui, QtCore


# external class for selecting layer mode:
from cdControlModeSelectToolBar import CDControlModeSelectToolBar

# external class for selecting layer mode:
from cdControlSceneZoomToolbar import CDControlSceneZoomToolbar

# external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ======================================================================
# 2012 - Mitja: CDToolBars is a class for grouping and providing access
#    to QToolBar elements in the application
# ======================================================================
class CDToolBars(QtCore.QObject):

    # ------------------------------------------------------------
    def __init__(self,pParent):
        # we don't use the pParent parameter to pass it to QObject, 
        #  but directly as parent object for all the QToolBar objects instantiated here:
        super(CDToolBars, self).__init__(None)

        self.mainWindow = pParent
#         
#         # demo toolbar object:
#         lOneToolbar = QtGui.QToolBar("QToolBar | OneToolbar")
#         CDConstants.printOut( "----- CDToolBars.__init__() - lOneToolbar = "+str(lOneToolbar), CDConstants.DebugExcessive )
#         # set all icons in this QToolBar to be of the same size:
#         lOneToolbar.setIconSize(QtCore.QSize(24, 24))
#         lOneToolbar.setObjectName("setObjectName OneToolbar")
#         lOneToolbar.setToolTip("setToolTip OneToolbar")
#         lOneToolbar.addWidget(QtGui.QLabel("*"))
#         lOneToolbar.addWidget(QtGui.QLabel(" "))
#         lOneToolbar.addWidget(QtGui.QLabel("_"))
#         lPrintOut = self.mainWindow.addToolBar(QtCore.Qt.TopToolBarArea, lOneToolbar)
#         CDConstants.printOut( "----- CDToolBars.__init__() - self.mainWindow.addToolBar(QtCore.Qt.TopToolBarArea, lOneToolbar) = "+str(lPrintOut), CDConstants.DebugExcessive )
#         # addToolBarBreak() places the next toolbar in the same toolbar area in a "new line"
#         # print self.mainWindow.addToolBarBreak(QtCore.Qt.TopToolBarArea)
# 

        # ----------
        # 2011 - Mitja: to control the "layer selection" for Cell Scene mode,
        #   we add a set of radio-buttons:
        self.__theModeSelectToolBar = CDControlModeSelectToolBar("CellDraw | Main Mode ToolBar")
        self.__theModeSelectToolBar.setObjectName("theModeSelectToolBar")
        lPrintOut = self.mainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.__theModeSelectToolBar)
        CDConstants.printOut( "----- CDToolBars.__init__() - self.mainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.__theModeSelectToolBar) = "+str(lPrintOut), CDConstants.DebugExcessive )
#         print self.mainWindow.addToolBarBreak(QtCore.Qt.TopToolBarArea)

        # ----------
        # 2012 - Mitja: to control the "scene zoom" factor,
        #   we add a "combo box":
        self.__theSceneZoomToolbar = CDControlSceneZoomToolbar("CellDraw | Scene Zoom ToolBar")
        self.__theSceneZoomToolbar.setObjectName("theSceneZoomToolbar")
        lPrintOut = self.mainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.__theSceneZoomToolbar)
        CDConstants.printOut( "----- CDToolBars.__init__() - self.mainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.__theSceneZoomToolbar) = "+str(lPrintOut), CDConstants.DebugExcessive )
#         print self.mainWindow.addToolBarBreak(QtCore.Qt.TopToolBarArea)


        CDConstants.printOut( "----- CDToolBars.__init__(pParent=="+str(pParent)+") done. -----", CDConstants.DebugExcessive )

    # ------------------------------------------------------------------
    # ----- end of init() -----
    # ------------------------------------------------------------------

# 
# 
#     # ------------------------------------------------------------
#     # set a checked button in the QButtonGroup:
#     # ------------------------------------------------------------
#     def setSelectedSceneMode(self, pId):
#         self.selectedSceneMode = pId
#         self.sceneModeActionDict[self.selectedSceneMode].setChecked(True)
# 

# 
# 

# 
#     # ------------------------------------------------------------
#     # register a callback handler function for the
#     #   "pSignal()" signal:
#     # ------------------------------------------------------------
#     def registerSignalHandler(self, pSignal, pHandler):
#         pSignal.connect( pHandler )
# 


# 
#     # ------------------------------------------------------------
#     # return the ID of the only checked button in the QButtonGroup:
#     # ------------------------------------------------------------
#     def getSelectedSceneMode(self):
#         return self.selectedSceneMode
# 



    # ------------------------------------------------------------
    # register any callback handlers for specific toolbars:
    # ------------------------------------------------------------



    # ------------------------------------------------------------
    # CDControlModeSelectToolBar:
    # register the callback handler function
    #   for the __signalModeSelectToolbarChanged signal generated by CDControlModeSelectToolBar:
    # ------------------------------------------------------------
    def registerHandlerForModeSelectToolbarControllerSignals(self, pHandler):
        self.__theModeSelectToolBar.registerHandlerForToolbarChanges( pHandler )

    # ------------------------------------------------------------
    # CDControlModeSelectToolBar:
    # callback handler function
    #   for the __signalModeSelectToolbarChanged signal generated by CDControlModeSelectToolBar:
    # ------------------------------------------------------------
    def handlerForChangeInGlobalModeModelSignals(self, pMode):
        self.__theModeSelectToolBar.setSelectedSceneMode(pMode)



    # ------------------------------------------------------------
    # CDControlModeSelectToolBar:
    # set the icon of the Image Layer selection button
    # ------------------------------------------------------------
    def setModeSelectToolBarImageLayerButtonIconFromPixmap(self, pPixmap):
        pass # 154 prrint "=-=-=-= CDToolBars.setModeSelectToolBarImageLayerButtonIconFromPixmap( pPixmap=="+str(pPixmap)+", isNull()=="+str(pPixmap.isNull())+" )"
        self.__theModeSelectToolBar.setImageLayerButtonIconFromPixmap( pPixmap )



    # ------------------------------------------------------------
    # CDControlModeSelectToolBar:
    # set one button of the mode select toolbar to be enabled or not enabled
    # ------------------------------------------------------------
    def setModeSelectToolBarButtonEnabled(self, pMode, pEnable=True):
        self.__theModeSelectToolBar.setEnabled( pMode, pEnable )



    # ------------------------------------------------------------
    # CDControlSceneZoomToolbar:
    # register the callback handler function
    #   for the signalZoomFactorHasChanged signal generated by CDControlSceneZoomToolbar:
    # ------------------------------------------------------------
    def registerHandlerForSceneZoomChangedControllerSignals(self, pHandler):
        self.__theSceneZoomToolbar.registerHandlerForToolbarChanges( pHandler )



# end class CDToolBars(QtGui.QObject)
# ======================================================================



# ------------------------------------------------------------
# just for testing:
# ------------------------------------------------------------
if __name__ == '__main__':
    CDConstants.printOut( "----- CDToolBars.__main__() 1", CDConstants.DebugAll )
    import sys

    app = QtGui.QApplication(sys.argv)

    CDConstants.printOut( "----- CDToolBars.__main__() 2", CDConstants.DebugAll )

    testQMainWindow = QtGui.QMainWindow()
    testQMainWindow.setGeometry(100, 100, 900, 500)
    testQMainWindow.setUnifiedTitleAndToolBarOnMac(False)

    CDConstants.printOut( "----- CDToolBars.__main__() 3", CDConstants.DebugAll )

    CDConstants.printOut( "testQMainWindow = "+str(testQMainWindow), CDConstants.DebugAll )

    lTestToolBarsObject = CDToolBars(testQMainWindow)

    CDConstants.printOut( "lTestToolBarsObject = "+str(lTestToolBarsObject), CDConstants.DebugAll )
   
#     print "NOW testQMainWindow.addToolBarsToMainWindow() ..."
#     print lTestToolBarsObject.addToolBarsToMainWindow()

    CDConstants.printOut( "----- CDToolBars.__main__() 4", CDConstants.DebugAll )

    # 2010 - Mitja: QMainWindow.raise_() must be called after QMainWindow.show()
    #     otherwise the PyQt/Qt-based GUI won't receive foreground focus.
    #     It's a workaround for a well-known bug caused by PyQt/Qt on Mac OS X
    #     as shown here:
    #       http://www.riverbankcomputing.com/pipermail/pyqt/2009-September/024509.html
    testQMainWindow.raise_()
    testQMainWindow.show()

    CDConstants.printOut( "----- CDToolBars.__main__() 5", CDConstants.DebugAll )

    sys.exit(app.exec_())

    CDConstants.printOut( "----- CDToolBars.__main__() 6", CDConstants.DebugAll )

# Local Variables:
# coding: US-ASCII
# End:
